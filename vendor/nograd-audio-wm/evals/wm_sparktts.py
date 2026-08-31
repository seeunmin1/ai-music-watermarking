import os
import sys
import json
import pickle
import random
import argparse
from pathlib import Path
import tempfile

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import special
import torchaudio
import re

# Make local clone importable (matches audio_processing_pipeline.py)
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)  # For watermark.engine imports
sys.path.insert(0, os.path.join(REPO_ROOT, "models", "Spark-TTS"))

from sparktts.models.audio_tokenizer import BiCodecTokenizer  # SparkTTS tokenizer
from cli.SparkTTS import SparkTTS
import sphn

from evals.main_wm import load_clustering_maps as load_general_clustering_maps
from models.moshi.utils import bool_inst
from training import get_validation_augs, get_dummy_augs

# ======================
# Utility
# ======================

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_binomial_pval(x, n, p):
    return special.betainc(x, 1 + n - x, p)

GENERATOR = torch.Generator()

def get_wm_window_hash(ngrams: torch.Tensor, wm_seed: int, clustering_map=None):
    if ngrams.numel() == 0:
        return torch.tensor([wm_seed], device=ngrams.device, dtype=torch.long)
    return (ngrams.to(torch.long).sum(dim=1) + wm_seed)

def compute_watermark_scores(wm_stream, ngrams, audio_vocab_size, gamma, wm_seed, clustering_map=None):
    seen_tokens = set()
    green_mask = torch.zeros_like(wm_stream, dtype=torch.bool)
    to_score_mask = torch.zeros_like(wm_stream, dtype=torch.bool)

    effective_vocab_size = int(clustering_map.max().item()) + 1 if clustering_map is not None else audio_vocab_size
    T = wm_stream.shape[-1]

    per_timestep_hashes = None
    if ngrams is not None and ngrams.numel() > 0 and ngrams.shape[0] == T:
        per_timestep_hashes = get_wm_window_hash(ngrams, wm_seed, clustering_map=clustering_map).cpu()

    single_hash_val = None
    if per_timestep_hashes is None:
        single_hash = get_wm_window_hash(ngrams, wm_seed, clustering_map=clustering_map)
        single_hash_val = int(single_hash[0].item())

    debug_checked = False
    for ii, token in enumerate(wm_stream):
        seed = int(per_timestep_hashes[ii].item()) if per_timestep_hashes is not None else single_hash_val
        GENERATOR.manual_seed(seed)
        vocab_perm = torch.randperm(effective_vocab_size, generator=GENERATOR)
        greenlist = vocab_perm[:int(gamma * effective_vocab_size)]

        token_val = token.cpu().item()
        if clustering_map is not None:
            cluster_id = clustering_map[token.long()].item()
            is_green = cluster_id in greenlist
            # Debug: check that cluster_id is in the candidate vocab
            assert 0 <= cluster_id < effective_vocab_size, f"Token {token_val} maps to cluster {cluster_id}, outside candidate vocab [0,{effective_vocab_size})"
            if not debug_checked:
                # Check that all cluster_ids in this stream are in candidate vocab
                all_clusters = clustering_map[wm_stream.long()].cpu().numpy()
                assert all((0 <= c < effective_vocab_size) for c in all_clusters), f"Some cluster ids in stream are outside candidate vocab [0,{effective_vocab_size})"
                debug_checked = True
        else:
            is_green = token_val in greenlist
            assert 0 <= token_val < effective_vocab_size, f"Token {token_val} outside candidate vocab [0,{effective_vocab_size})"

        green_mask[ii] = is_green
        if token_val not in seen_tokens:
            to_score_mask[ii] = 1
            seen_tokens.add(token_val)

    return green_mask, to_score_mask

def build_stream_ngrams_from_full_stream(stream_tokens: torch.Tensor, wm_ngram: int, device='cpu'):
    T = int(stream_tokens.shape[-1])
    n = int(wm_ngram)
    if n <= 0:
        return torch.zeros((T, 0), dtype=torch.long, device=device)
    s = stream_tokens.to(torch.long).to(device)
    rows = []
    for i in range(T):
        start = max(0, i - n)
        ctx = s[start:i]
        if ctx.shape[-1] < n:
            pad = torch.zeros((n - ctx.shape[-1],), dtype=torch.long, device=device)
            ctx = torch.cat([pad, ctx], dim=0)
        rows.append(ctx.unsqueeze(0))
    return torch.cat(rows, dim=0)

# ======================
# Config loaders (kept)
# ======================

# def load_configs_general(args):
#     configs = []
#     all_maps = load_general_clustering_maps(args.clustering_dir, target_min_count=None, device=args.device)
#     if all_maps and 0 in all_maps:
#         for m in list(all_maps[0].keys()):
#             for key in sorted(list(all_maps[0][m].keys())):
#                 current_config_maps = {}
#                 valid_config = True
#                 for s in args.wm_streams:
#                     c = int(s)
#                     if c in all_maps and m in all_maps[c] and key in all_maps[c][m]:
#                         _, tmap = all_maps[c][m][key]
#                         current_config_maps[c] = tmap
#                     else:
#                         valid_config = False
#                         break
#                 if valid_config:
#                     configs.append({"method": m, "min_count": key, "maps": current_config_maps, "name": f"{m}_{key}"})
#     return configs

def load_configs_ablate(args):
    configs = []
    with open(args.clustering_pkl, "rb") as f:
        clusterings = pickle.load(f)

    keys = sorted(clusterings[0].keys())
    vocab_size = 8192
    for key in keys:
        cnt, res = key
        current_config_maps = {}
        for s in args.wm_streams:
            stream_id = int(s)
            cmap_tensor = torch.as_tensor(clusterings[stream_id][key], device=args.device, dtype=torch.long)
            full_map = torch.full((vocab_size,), -1, device=args.device, dtype=torch.long)
            limit = min(len(cmap_tensor), vocab_size)
            full_map[:limit] = cmap_tensor[:limit]
            unmapped = (full_map == -1)
            if unmapped.any():
                k_clusters = int(cmap_tensor.max().item()) + 1
                full_map[unmapped] = torch.arange(k_clusters, k_clusters + unmapped.sum(), device=args.device)
            current_config_maps[stream_id] = full_map
        configs.append({"method": "leiden", "min_count": int(cnt), "res": float(res), "maps": current_config_maps, "name": f"leiden_{cnt}_res{res}"})
    return configs

def load_configs_select(args):
    configs = []
    with open(args.clustering_pkl, "rb") as f:
        data = pickle.load(f)
    vocab_size = 8192
    selected_count = 10
    selected_resolution = 1.0
    current_config_maps = {}
    for s in args.wm_streams:
        s_int = 0
        pkl_key = 0
        cmap = torch.as_tensor(data[pkl_key][(selected_count, selected_resolution)], device=args.device, dtype=torch.long)
        full_map = torch.full((vocab_size,), -1, device=args.device, dtype=torch.long)
        limit = min(len(cmap), vocab_size)
        full_map[:limit] = cmap[:limit]
        unmapped = (full_map == -1)
        if unmapped.any():
            start_id = int(cmap.max().item()) + 1
            full_map[unmapped] = torch.arange(start_id, start_id + unmapped.sum(), device=args.device)
        current_config_maps[s_int] = full_map
    configs.append({"method": "selected", "maps": current_config_maps, "name": "selected"})
    return configs

# ======================
# SparkTTS Tokenizer wrapper (unchanged IO)
# ======================

class SparkTTSTokenizerWrapper:
    def __init__(self, model_dir: str, device: torch.device):
        self.tok = BiCodecTokenizer(model_dir=model_dir, device=device)
        self.sr = int(self.tok.config.get("sample_rate", 16000))
        self.device = device

    def encode_semantic(self, wav: torch.Tensor) -> torch.Tensor:
        wav_cpu = wav.detach().cpu()
        if wav_cpu.dim() == 3 and wav_cpu.shape[0] == 1:  # [1, C, T]
            wav_cpu = wav_cpu.squeeze(0)
        elif wav_cpu.dim() == 1:  # [T]
            wav_cpu = wav_cpu.unsqueeze(0)  # -> [1, T]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, wav_cpu, self.sr)
            path = tmp.name
        try:
            _, s = self.tok.tokenize(path)   # returns (1,1,32) global, (1,T) semantic
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return s.squeeze(0).to(self.device)

    def decode(self, semantic: torch.Tensor, global_tokens: torch.Tensor = None) -> torch.Tensor:
        s = semantic.view(1, -1).cpu()
        if global_tokens is None:
            g = torch.zeros(1, 1, 32, dtype=torch.float32)
        else:
            g = global_tokens.view(1, 1, -1).cpu()
        with torch.no_grad():
            wav = self.tok.detokenize(g, s)
        return torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)

# ======================
# Core eval
# ======================

def run_watermark_eval(args, clustering_maps=None, config_name="base"):
    device = args.device
    sparktts = SparkTTS(model_dir=Path(args.sparktts_model_dir), device=args.device)
    tokenizer = SparkTTSTokenizerWrapper(args.sparktts_model_dir, device)
    sr = tokenizer.sr
    semantic_vocab_size = int(getattr(sparktts, "semantic_vocab_size", 0))
    assert semantic_vocab_size == 8192

    # Prompts
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if args.nsamples > 0:
        prompts = prompts[:args.nsamples]

    # Augs
    frame_size = int(0.08 * sr)  # keep augmentation timing at ~80 ms across sample rates
    augs = get_validation_augs(sample_rate=sr, frame_size=frame_size) if args.eval_aug else get_dummy_augs()
    for aug, _ in augs:
        aug.to(device)

    global_results = []
    os.makedirs(args.output_dir, exist_ok=True)
    generated_text_path = os.path.join(args.output_dir, f"generated_texts_{config_name}.txt")
    # Reset references per run so WER scripts don't mix stale prompts from previous executions.
    with open(generated_text_path, "w", encoding="utf-8"):
        pass

    for batch_start in tqdm(range(0, len(prompts), args.batch_size)):
        batch_texts = prompts[batch_start: batch_start + args.batch_size]

        batch_audio = []
        batch_tokens = []
        for txt in batch_texts:
            if args.steps > 0:
                max_new_tokens = args.steps
            else:
                max_new_tokens = args.sparktts_max_new_tokens
            if not args.allow_short_steps and max_new_tokens < args.sparktts_min_new_tokens:
                print(
                    f"[WARN] max_new_tokens={max_new_tokens} is very low for SparkTTS and often truncates speech; "
                    f"using {args.sparktts_min_new_tokens} instead. Set --allow_short_steps True to keep short generations.",
                    flush=True,
                )
                max_new_tokens = args.sparktts_min_new_tokens

            wav, sem_gen = sparktts.inference(
                text=txt,
                prompt_speech_path=Path(args.sparktts_prompt_audio) if args.sparktts_prompt_audio else None,
                prompt_text=args.sparktts_prompt_text,
                temperature=args.temperature,
                top_k=args.sparktts_top_k,
                top_p=args.sparktts_top_p,
                max_new_tokens=max_new_tokens,
                wm_enable=(args.wm_method != "none"),
                wm_method=args.wm_method,
                wm_gamma=args.wm_gamma,
                wm_delta=args.wm_delta,
                wm_ngram=args.wm_ngram,
                wm_seed=args.wm_seed,
                clustering_map=clustering_maps.get(0) if clustering_maps else None,
                return_semantic_ids=True,
            )
            wav = torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
            batch_audio.append(wav)
            # Original tokens are model-generated semantic IDs (pre waveform roundtrip).
            batch_tokens.append(sem_gen.to(device))

        batch_audio = torch.stack(batch_audio, dim=0)  # [B,1,T]

        for validation_aug, strengths in augs:
            for strength in strengths:
                batch_aug_audio, _ = validation_aug(batch_audio.clone(), None, strength)

                for i in range(batch_aug_audio.shape[0]):
                    synced_audio = batch_aug_audio[i:i+1]
                    tokens_rt = tokenizer.encode_semantic(synced_audio)
                    # Keep original semantic stream unpadded for fair token-level comparison.
                    orig_tokens = batch_tokens[i]

                    # n-grams (single stream)
                    ngrams_orig = build_stream_ngrams_from_full_stream(orig_tokens, args.wm_ngram, device='cpu')
                    ngrams_rt   = build_stream_ngrams_from_full_stream(tokens_rt, args.wm_ngram, device='cpu')

                    s_map = clustering_maps.get(0) if clustering_maps else None
                    vocab_size_orig = semantic_vocab_size
                    vocab_size_rt = semantic_vocab_size

                    g_mask_o, s_mask_o = compute_watermark_scores(
                        orig_tokens, ngrams_orig, vocab_size_orig, args.wm_gamma, args.wm_seed, clustering_map=s_map
                    )
                    g_mask_r, s_mask_r = compute_watermark_scores(
                        tokens_rt, ngrams_rt, vocab_size_rt, args.wm_gamma, args.wm_seed, clustering_map=s_map
                    )

                    greens_o = float((g_mask_o * s_mask_o).float().sum().item())
                    greens_r = float((g_mask_r * s_mask_r).float().sum().item())
                    scored_o = float(s_mask_o.float().sum().item())
                    scored_r = float(s_mask_r.float().sum().item())

                    pval_o = get_binomial_pval(greens_o, scored_o, args.wm_gamma)
                    pval_r = get_binomial_pval(greens_r, scored_r, args.wm_gamma)
                    global_idx = batch_start + i

                    global_results.append({
                        "config": config_name,
                        "idx": global_idx,
                        "prompt": batch_texts[i],
                        "aug_name": str(validation_aug),
                        "strength": strength,
                        "greens_orig": greens_o,
                        "scored_orig": scored_o,
                        "greens_rt": greens_r,
                        "scored_rt": scored_r,
                        "pval_orig": pval_o,
                        "pval_rt": pval_r,
                        "ntoks_orig": int(orig_tokens.numel()),
                        "ntoks_rt": int(tokens_rt.numel()),
                    })

                    if args.save_audio > 0 and global_idx < args.save_audio:
                        audio_output_dir = os.path.join(args.output_dir, f"audio_{config_name}")
                        os.makedirs(audio_output_dir, exist_ok=True)
                        sphn.write_wav(
                            os.path.join(audio_output_dir, f'{validation_aug}_{strength}_{global_idx:03d}.wav'),
                            batch_aug_audio[i, 0].detach().cpu().numpy().astype(np.float32),
                            sr,
                        )

                    tok_dir = os.path.join(args.output_dir, "tokens", f"{config_name}")
                    os.makedirs(tok_dir, exist_ok=True)

                    aug_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(validation_aug))
                    strength_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(strength))
                    token_prefix = f"{global_idx:04d}_{aug_tag}_{strength_tag}"

                    with open(os.path.join(tok_dir, f"{token_prefix}_orig.txt"), "w") as f:
                        f.write(" ".join(map(str, orig_tokens.tolist())) + "\n")
                    with open(os.path.join(tok_dir, f"{token_prefix}_rt.txt"), "w") as f:
                        f.write(" ".join(map(str, tokens_rt.tolist())) + "\n")

        with open(generated_text_path, "a", encoding="utf-8") as f:
            for idx, txt in enumerate(batch_texts):
                f.write(f"{batch_start + idx:04d},{txt}\n")

    torch.save({'config': vars(args), 'results': global_results}, os.path.join(args.output_dir, f'summary_{config_name}.pt'))

    df = pd.DataFrame([{
        "idx": r["idx"], "aug_name": r["aug_name"], "strength": str(r["strength"]),
        "greens_orig": r["greens_orig"], "scored_orig": r["scored_orig"],
        "greens_rt": r["greens_rt"], "scored_rt": r["scored_rt"],
        "ntoks_orig": r["ntoks_orig"], "ntoks_rt": r["ntoks_rt"],
        "pval_orig": r["pval_orig"], "pval_rt": r["pval_rt"],
        "logp_orig": -np.log10(r["pval_orig"]) if r["pval_orig"] is not None and r["pval_orig"] > 0 else None,
        "logp_rt": -np.log10(r["pval_rt"]) if r["pval_rt"] is not None and r["pval_rt"] > 0 else None,
    } for r in global_results])

    cols = [c for c in ["greens_orig","scored_orig","greens_rt","scored_rt","ntoks_orig","ntoks_rt","pval_orig","pval_rt","logp_orig","logp_rt"] if c in df.columns]
    mean_df = df.groupby(["aug_name", "strength"])[cols].agg("mean")
    mean_df.to_csv(os.path.join(args.output_dir, f"summary_{config_name}.csv"))

    df.to_csv(os.path.join(args.output_dir, f"results_{config_name}.csv"), index=False)

# ======================
# Main
# ======================

def main():
    parser = argparse.ArgumentParser(description="SparkTTS Watermark Evaluation Pipeline")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.device_count() else "cpu")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to txt file with prompts")
    parser.add_argument("--nsamples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--wm_method", type=str, default="maryland")
    parser.add_argument("--wm_streams", nargs='+', default=[0], help="SparkTTS uses stream 0")
    parser.add_argument("--wm_delta", type=float, default=2.0)
    parser.add_argument("--wm_gamma", type=float, default=0.25)
    parser.add_argument("--wm_ngram", type=int, default=0)
    parser.add_argument("--wm_seed", type=int, default=0)
    parser.add_argument("--encodec_weight", type=str, default=None)
    parser.add_argument("--save_audio", type=int, default=10)
    parser.add_argument("--eval_aug", type=bool_inst, default=True)

    parser.add_argument("--run_mode", type=str, choices=["base", "general", "ablate", "select"], default="base")
    parser.add_argument("--clustering_dir", type=str, default="models/embeddings/clusterings")
    parser.add_argument("--clustering_pkl", type=str, default="models/embeddings/clusterings_sparktts/leiden_clusterings_trainonly_allparams.pkl")

    parser.add_argument("--sparktts_model_dir", type=str, default=None,
                        help="Path to a local Spark-TTS checkpoint directory.")
    parser.add_argument("--sparktts_prompt_audio", type=str, default=None, help="Path to reference audio for voice cloning")
    parser.add_argument("--sparktts_prompt_text", type=str, default=None, help="Transcript of the prompt audio")
    parser.add_argument("--sparktts_top_k", type=int, default=50)
    parser.add_argument("--sparktts_top_p", type=float, default=0.95)
    parser.add_argument("--sparktts_max_new_tokens", type=int, default=3000)
    parser.add_argument("--sparktts_min_new_tokens", type=int, default=800)
    parser.add_argument("--allow_short_steps", type=bool_inst, default=False)

    args = parser.parse_args()
    args.device = torch.device(args.device)
    seed_all(args.seed)

    if not args.sparktts_model_dir:
        parser.error("--sparktts_model_dir is required for Spark-TTS runs.")

    if args.sparktts_prompt_audio and args.sparktts_prompt_text is None:
        print(
            "[WARN] --sparktts_prompt_audio was provided without --sparktts_prompt_text. "
            "This can reduce voice-cloning quality and may worsen WER.",
            flush=True,
        )

    if args.run_mode == "base":
        configs_to_run = [{"method": None, "maps": None, "name": "base"}]
    # elif args.run_mode == "general":
    #     configs_to_run = load_configs_general(args)
    elif args.run_mode == "ablate":
        configs_to_run = load_configs_ablate(args)
    elif args.run_mode == "select":
        configs_to_run = load_configs_select(args)
    else:
        configs_to_run = [{"method": None, "maps": None, "name": "base"}]

    if not configs_to_run:
        print("No configurations generated. Running fallback base.")
        configs_to_run = [{"method": None, "maps": None, "name": "base"}]

    print(f"Starting execution for {len(configs_to_run)} configuration(s).")
    for config in configs_to_run:
        print("\nRUNNING FOR:", config["name"])
        run_watermark_eval(args, clustering_maps=config["maps"], config_name=config["name"])

if __name__ == "__main__":
    main()
