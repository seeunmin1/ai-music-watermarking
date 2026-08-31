import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
import sphn
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    EncodecModel,
    MusicgenForConditionalGeneration,
)

from evals.main_wm import get_binomial_pval, seed_all, load_clustering_maps, compute_watermark_scores

from models.moshi.utils import bool_inst
from models.musicgen import MusicGenWMGen

from training import get_validation_augs, get_dummy_augs


def build_stream_ngrams_from_full_stream(stream_tokens: torch.Tensor, wm_ngram: int, device='cpu'):
    """
    Constructs n-grams for detection.
    CRITICAL FIX: Uses strictly previous tokens [i-n : i] as context for token i.
    """
    T = int(stream_tokens.shape[-1])
    n = int(wm_ngram)
    # If 0-gram, return empty contexts
    if n <= 0:
        return torch.zeros((T, 0), dtype=torch.long, device=device)
    
    s = stream_tokens.to(torch.long).to(device)
    rows = []
    for i in range(T):
        # Context ends at i (exclusive), so it does not include the current token
        start = max(0, i - n)
        ctx = s[start : i]
        
        # Left-pad with zeros if context is shorter than n (at the beginning of stream)
        L = ctx.shape[-1]
        if L < n:
            pad = torch.zeros((n - L,), dtype=torch.long, device=device)
            ctx = torch.cat([pad, ctx], dim=0)
        rows.append(ctx.unsqueeze(0))
        
    return torch.cat(rows, dim=0)


def run_watermark_eval(args, clustering_maps=None, config_name="base"):
    """Generate audio with watermarks and evaluate watermark preservation"""
    # 1. Load Models (Swapped for MusicGen)
    device = args.device
    print(f"Loading MusicGen models on {device}...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium").to(device)
    encodec = EncodecModel.from_pretrained("facebook/encodec_32khz").to(device)

    # 2. Apply Weight Translation (from musicgen.py)
    if args.encodec_weight:
        print(f"Loading finetuned EnCodec weights from {args.encodec_weight}")
        raw_sd = torch.load(args.encodec_weight, map_location=device)
        if "model_state" in raw_sd: raw_sd = raw_sd["model_state"]
        translated_sd = {}
        for k, v in raw_sd.items():
            nk = k.replace("encoder.model", "encoder.layers").replace("decoder.model", "decoder.layers").replace("quantizer.vq", "quantizer.layers").replace("conv.conv.", "conv.")
            translated_sd[nk] = v
        encodec.load_state_dict(translated_sd, strict=False)
        model.audio_encoder = encodec # Ensure MusicGen uses the swapped encoder

    # 3. Initialize Wrapper
    lm_gen = MusicGenWMGen(
        model, 
        temp=args.temperature, 
        wm=args.wm_method, 
        wm_ngram=args.wm_ngram,
        wm_seed=args.wm_seed, 
        wm_streams=[int(s) for s in args.wm_streams],
        wm_aux_params={"delta": args.wm_delta, "gamma": args.wm_gamma, "clustering_maps": clustering_maps}
    )

    print(f"Watermarking config: method={lm_gen.wm}, streams={lm_gen.wm_streams}, "
          f"ngram={lm_gen.wm_ngram}, delta={lm_gen.wm_aux_params['delta']}")
    print(f"--- Running Configuration: {config_name} ---")

    # 4. Handle Prompts (Text File)
    with open(args.prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    nsamples = len(prompts)
    if args.nsamples > 0:
        nsamples = min(args.nsamples, nsamples)
        prompts = prompts[:nsamples]

    global_watermark_results = []
    
    # Loop over samples in batches
    for batch_start in tqdm(range(0, nsamples, args.batch_size)):
        batch_size = min(args.batch_size, nsamples - batch_start)
        batch_texts = prompts[batch_start : batch_start + batch_size]
        
        inputs = processor(text=batch_texts, padding=True, return_tensors="pt").to(device)

        # 5. Generate
        wm_tokens_th = lm_gen.generate_watermarked(inputs, max_new_tokens=args.steps) # [B, K, T]
        
        # Decode to audio using the EnCodec model
        with torch.no_grad():
            # wm_tokens_th shape: [B, K, T]
            # EnCodec.decode expects [B, K, T] and returns a specific output object
            decoded_outputs = encodec.decode(wm_tokens_th[None, :], [None] * batch_size)
            batch_all_audio = decoded_outputs.audio_values # [B, 1, L]

        # 6. Evaluation Loop
        augs = get_validation_augs() if args.eval_aug else get_dummy_augs()
        for aug, _ in augs:
            aug.to(args.device)
        
        batch_audio_saved = batch_all_audio.clone()
        
        for validation_aug, strengths in augs:
            for strength in strengths:
                batch_aug_audio, _ = validation_aug(batch_audio_saved, None, strength)

                for idx in range(batch_size):
                    synced_audio = batch_aug_audio[idx:idx+1]

                    # Encode augmented audio (Roundtrip)
                    # MusicGen/Encodec: [1, 4, T]
                    tokens_roundtrip = encodec.encode(synced_audio).audio_codes.squeeze(0).squeeze(0)

                    # Get watermarked streams (Original)
                    # Use index slicing directly (0-3)
                    wm_tokens_orig = wm_tokens_th[idx] # [K, T]

                    orig_greens, orig_scored = [], []
                    greens, scored = [], []

                    # Analyze Codebooks
                    # Note: We use self-history for all streams in MusicGen parallel generation
                    for s_idx, stream_id in enumerate(args.wm_streams):
                        stream_id = int(stream_id)
                        
                        # A. Original Scores
                        wm_stream = wm_tokens_orig[stream_id, :]
                        ngrams_orig = build_stream_ngrams_from_full_stream(wm_stream, args.wm_ngram, device='cpu')
                        s_map = clustering_maps.get(stream_id) if clustering_maps else None
                        
                        g_mask, s_mask = compute_watermark_scores(
                            wm_stream, ngrams_orig, 2048, args.wm_gamma, args.wm_seed, clustering_map=s_map
                        )
                        orig_greens.append((g_mask * s_mask).float().sum().item())
                        orig_scored.append(s_mask.float().sum().item())

                        # B. Roundtrip Scores
                        if tokens_roundtrip is not None and stream_id < tokens_roundtrip.shape[0]:
                            wm_stream_rt = tokens_roundtrip[stream_id, :]
                            ngrams_rt = build_stream_ngrams_from_full_stream(wm_stream_rt, args.wm_ngram, device='cpu')
                            
                            g_mask_rt, s_mask_rt = compute_watermark_scores(
                                wm_stream_rt, ngrams_rt, 2048, args.wm_gamma, args.wm_seed, clustering_map=s_map
                            )
                            greens.append((g_mask_rt * s_mask_rt).float().sum().item())
                            scored.append(s_mask_rt.float().sum().item())
                        else:
                            greens.append(0)
                            scored.append(0)

                    # Calculate Stats
                    tot_orig_greens = float(sum(orig_greens))
                    tot_orig_scored = float(sum(orig_scored))
                    orig_pval = get_binomial_pval(tot_orig_greens, tot_orig_scored, args.wm_gamma)
                    
                    tot_greens = sum(greens)
                    tot_scored = sum(scored)
                    pval = get_binomial_pval(tot_greens, tot_scored, args.wm_gamma)
                    
                    global_idx = batch_start + idx
                    result = {
                        "config": config_name,
                        "idx": global_idx,
                        "aug_name": str(validation_aug),
                        "strength": strength,
                        "original_greens": orig_greens,
                        "original_ntoks": wm_tokens_orig.shape[-1],
                        "original_pval": orig_pval,
                        "greens": greens,
                        "scored": scored,
                        "ntoks": tokens_roundtrip.shape[-1],
                        "pval": pval,
                    }
                    global_watermark_results.append(result)

                    # Save generated audio
                    if args.save_audio > 0 and global_idx < args.save_audio:
                        audio_output_dir = os.path.join(args.output_dir, f"audio_{config_name}")
                        os.makedirs(audio_output_dir, exist_ok=True)
                        aug_audio_np = batch_aug_audio[idx, 0].detach().cpu().numpy().astype(np.float32)
                        sphn.write_wav(
                            os.path.join(audio_output_dir, f'{validation_aug}_{strength}_{global_idx:03d}.wav'),
                            aug_audio_np, encodec.config.sampling_rate
                        )

        # Save Text Prompts
        with open(os.path.join(args.output_dir, f"generated_texts_{config_name}.txt"), "a", encoding="utf-8") as f:
            for idx in range(batch_size):
                f.write(f"{idx + batch_start:04d},{batch_texts[idx]}\n")

    # Save summary
    summary = {'config': vars(args), 'results': global_watermark_results}
    torch.save(summary, os.path.join(args.output_dir, f'summary_{config_name}.pt'))

    # Calculate statistics
    if not global_watermark_results:
        print("No watermark results were produced for this configuration.")
        empty_df = pd.DataFrame(columns=["idx", "aug_name", "strength", "greens", "scored", "ntoks", "pval", "logpval"])
        empty_df.to_csv(os.path.join(args.output_dir, f'summary_{config_name}.csv'))
        empty_df.to_csv(os.path.join(args.output_dir, f'results_{config_name}.csv'), index=False)
        return

    df_data = [
        {
            "idx": wmr["idx"],
            "aug_name": str(wmr.get("aug_name", "identity")),
            "strength": str(wmr.get("strength", 0.0)),
            "greens": sum(wmr["greens"]),
            "scored": sum(wmr["scored"]),
            "ntoks": wmr["ntoks"],
            "pval": wmr["pval"],
            "logpval": -np.log10(wmr["pval"]) if wmr["pval"] is not None and wmr["pval"] > 0 else None,
        }
        for wmr in global_watermark_results
    ]

    df = pd.DataFrame(df_data)
    numeric_cols_for_mean = ["greens", "scored", "ntoks", "pval", "logpval"]
    cols_to_aggregate = [col for col in numeric_cols_for_mean if col in df.columns]
    
    mean_df = df.groupby(["aug_name", "strength"])[cols_to_aggregate].agg("mean")
    mean_df.to_csv(os.path.join(args.output_dir, f'summary_{config_name}.csv'))
    
    df.to_csv(os.path.join(args.output_dir, f'results_{config_name}.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.device_count() else "cpu")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to txt file with prompts")
    parser.add_argument("--nsamples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--wm_method", type=str, default="maryland")
    parser.add_argument("--wm_streams", nargs='+', default=[0, 1, 2, 3], help="Stream indices (0-3 for MusicGen)")
    parser.add_argument("--wm_delta", type=float, default=2.0)
    parser.add_argument("--wm_gamma", type=float, default=0.25)
    parser.add_argument("--wm_ngram", type=int, default=0)
    parser.add_argument("--wm_seed", type=int, default=0)
    parser.add_argument("--wm_clustering", type=bool_inst, default=False)
    parser.add_argument(
        "--clustering_pkl",
        type=str,
        default="models/embeddings/encodec_leiden_clusterings_trainonly_allparams.pkl",
        help="Path to the EnCodec clustering pickle used for clustered watermarking.",
    )
    parser.add_argument(
        "--clustering_select_path",
        type=str,
        default="models/encodec_configs.json",
        help="Optional JSON file selecting one clustering per EnCodec codebook.",
    )
    parser.add_argument("--encodec_weight", type=str, default=None)
    parser.add_argument("--save_audio", type=int, default=10)
    parser.add_argument("--eval_aug", type=bool_inst, default=True)         
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_all(args.seed)

    configs_to_run = []
    all_maps = None

    if args.wm_clustering:
        select_map = None
        if args.clustering_select_path and os.path.exists(args.clustering_select_path):
            with open(args.clustering_select_path, "r", encoding="utf-8") as f:
                select_map = json.load(f)

        all_maps = load_clustering_maps(
            args.clustering_pkl,
            select_map=select_map,
            device=args.device,
        )

        if select_map:
            current_config_maps = {}
            valid_config = True
            for s in args.wm_streams:
                c = int(s)
                if c not in all_maps:
                    valid_config = False
                    break
                method = next(iter(all_maps[c].keys()), None)
                if method is None:
                    valid_config = False
                    break
                key = next(iter(all_maps[c][method].keys()), None)
                if key is None:
                    valid_config = False
                    break
                _, tmap = all_maps[c][method][key]
                current_config_maps[int(s)] = tmap

            if valid_config:
                configs_to_run.append({"method": "clustered", "min_count": None, "maps": current_config_maps})
        elif all_maps and 0 in all_maps:
            found_methods = list(all_maps[0].keys())
            for m in found_methods:
                keys_available = sorted(list(all_maps[0][m].keys()))
                for key in keys_available:
                    current_config_maps = {}
                    valid_config = True
                    for s in args.wm_streams:
                        c = int(s)
                        if c in all_maps and m in all_maps[c] and key in all_maps[c][m]:
                            _, tmap = all_maps[c][m][key]
                            current_config_maps[int(s)] = tmap
                        else:
                            valid_config = False
                            break

                    if valid_config:
                        configs_to_run.append({
                            "method": m,
                            "min_count": key,
                            "maps": current_config_maps
                        })

    if not configs_to_run:
        configs_to_run = [{"method": None, "maps": None}]

    print(f"Starting execution for {len(configs_to_run)} configurations")

    for config in configs_to_run:
        if config["method"] is None:
            config_name = "base"
        elif config["min_count"] is None:
            config_name = config["method"]
        else:
            config_name = f"{config['method']}_{config['min_count']}"
        run_watermark_eval(args, clustering_maps=config["maps"], config_name=config_name)


if __name__ == "__main__":
    main()
