import os
import sys
import json
import pickle
import random
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import special
import torchaudio
import re

if not hasattr(torch, "xpu"):
    class _TorchXpuStub:
        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def is_available():
            return False

        @staticmethod
        def _is_in_bad_fork():
            return False

        @staticmethod
        def manual_seed_all(seed):
            return None

        @staticmethod
        def manual_seed(seed):
            return None

        @staticmethod
        def device_count():
            return 0

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    torch.xpu = _TorchXpuStub()

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

COSY_ROOT = os.path.join(REPO_ROOT, "models", "CosyVoice")
sys.path.insert(0, COSY_ROOT)
sys.path.insert(0, os.path.join(COSY_ROOT, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3
import sphn

from evals.main_wm import load_clustering_maps as load_general_clustering_maps
from models.moshi.utils import bool_inst
from training import get_validation_augs, get_dummy_augs

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

    for ii, token in enumerate(wm_stream):
        seed = int(per_timestep_hashes[ii].item()) if per_timestep_hashes is not None else single_hash_val
        GENERATOR.manual_seed(seed)
        vocab_perm = torch.randperm(effective_vocab_size, generator=GENERATOR)
        greenlist = vocab_perm[:int(gamma * effective_vocab_size)]

        token_val = token.cpu().item()
        if clustering_map is not None:
            cluster_id = clustering_map[token.long()].item()
            is_green = cluster_id in greenlist
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

def load_configs_ablate(args):
    configs = []
    with open(args.clustering_pkl, "rb") as f:
        clusterings = pickle.load(f)
    keys = sorted(clusterings[0].keys())
    vocab_size = 8192 # ?
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
    vocab_size = 8192 # ?
    selected_count = 1
    selected_resolution = 0.8
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

class CosyVoiceTokenizerWrapper:

    def __init__(self, model_dir: str, device: torch.device):
        import onnxruntime
        import whisper
        self.device = device
        self.model_dir = model_dir
        self.sr = 16000  # CosyVoice3 expects 16kHz input for tokenizer
        self.sr_out = 24000  # CosyVoice3 vocoder output is 24kHz
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = 1
        self.st_sess = onnxruntime.InferenceSession(
            os.path.join(model_dir, "speech_tokenizer_v3.onnx"),
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.whisper = whisper

    def encode_semantic(self, wav: torch.Tensor) -> torch.Tensor:
        # Accept shapes: [1, T], [B, 1, T], [B, T]; squash to [1, T]
        w = wav
        if w.dim() == 3 and w.shape[1] == 1:
            w = w.squeeze(1)
        if w.dim() == 2:
            w = w[0]
        w = w.cpu()
        # Resample to 16kHz if needed
        if hasattr(self, 'sr') and hasattr(self, 'sr_out') and self.sr != self.sr_out:
            import torchaudio
            w = torchaudio.functional.resample(w, self.sr_out, self.sr)
        mel = self.whisper.log_mel_spectrogram(w, n_mels=128)
        mel_np = mel.unsqueeze(0).numpy()
        codes = self.st_sess.run(
            None,
            {
                self.st_sess.get_inputs()[0].name: mel_np,
                self.st_sess.get_inputs()[1].name: np.array([mel_np.shape[2]], dtype=np.int32),
            },
        )[0][0]
        codes_t = torch.from_numpy(codes).to(wav.device).long()
        return codes_t

    # Why can't we get the original tokens? CosyVoice's TTS pipeline (inference_sft and CosyVoiceModel) does not expose or return the autoregressive/semantic tokens generated during synthesis. These tokens are only used internally to produce the waveform and are not saved or returned in the output. Unless the model is modified to return these tokens, we cannot access them from the outside.

    def decode(self, semantic: torch.Tensor) -> torch.Tensor:
        # Placeholder: implement decoding semantic tokens to wav using CosyVoice
        raise NotImplementedError("CosyVoice semantic token decoding not implemented.")

def run_watermark_eval(args, clustering_maps=None, config_name="base"):
    device = args.device
    # Use CosyVoice3 for CosyVoice3 models
    from cosyvoice.cli.cosyvoice import CosyVoice3
    cosyvoice = CosyVoice3(model_dir=Path(args.cosyvoice_model_dir))
    # CosyVoice yaml applies fixed seeds at load time; re-apply user seed so runs with
    # different --seed values actually diverge when sampling is enabled.
    seed_all(args.seed)
    tokenizer = CosyVoiceTokenizerWrapper(args.cosyvoice_model_dir, device)
    sr = tokenizer.sr
    semantic_vocab_size = 6561  # CosyVoice3 speech_token_size is 6561

    ASSISTANT_PREFIX = "You are a helpful assistant.<|endofprompt|>"

    def ensure_endofprompt_prefix(text: str) -> str:
        if "<|endofprompt|>" in text:
            return text
        return ASSISTANT_PREFIX + text


    # CosyVoice3 requires <|endofprompt|> token (id 151646) in the text
    END_OF_PROMPT = "<|endofprompt|>"
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    # Append <|endofprompt|> if not present
    prompts = [p if p.endswith(END_OF_PROMPT) else p + END_OF_PROMPT for p in prompts]
    if args.nsamples > 0:
        prompts = prompts[:args.nsamples]

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

    # Construct wm_args for watermarking
    wm_args = {
        'wm_method': args.wm_method,
        'wm_gamma': args.wm_gamma,
        'wm_delta': args.wm_delta,
        'wm_ngram': args.wm_ngram,
        'wm_seed': args.wm_seed,
        'clustering_map': clustering_maps.get(0) if clustering_maps else None,
        'temp': args.temperature,
        'top_k': 0,
        'top_p': 0.0,
        'use_sampling': True,
    }
    print(wm_args)

    for batch_start in tqdm(range(0, len(prompts), args.batch_size)):
        batch_texts = prompts[batch_start: batch_start + args.batch_size]
        batch_audio = []
        batch_tokens = []
        for txt in batch_texts:
            if args.prompt_wav:
                if args.prompt_text:
                    prompt_text = ensure_endofprompt_prefix(args.prompt_text)
                    model_input = cosyvoice.frontend.frontend_zero_shot(
                        txt,
                        prompt_text,
                        args.prompt_wav,
                        cosyvoice.sample_rate,
                        "",
                    )
                else:
                    txt = ensure_endofprompt_prefix(txt)
                    model_input = cosyvoice.frontend.frontend_cross_lingual(
                        txt,
                        args.prompt_wav,
                        cosyvoice.sample_rate,
                        "",
                    )
                for model_output in cosyvoice.model.tts(**model_input, stream=False, speed=1.0, return_token_sequence=True, wm_args=wm_args):
                    wav = torch.tensor(model_output['tts_speech'], dtype=torch.float32, device=device).unsqueeze(0)
                    batch_audio.append(wav)
                    ar_tokens = model_output.get('ar_tokens', None)
                    if ar_tokens is None:
                        raise RuntimeError('CosyVoice model did not return ar_tokens (original token sequence). This is required for watermark detection.')
                    ar_tokens = ar_tokens.squeeze(0) if ar_tokens.dim() == 2 and ar_tokens.shape[0] == 1 else ar_tokens
                    batch_tokens.append(ar_tokens)
                    break
            else:
                available_spks = cosyvoice.list_available_spks()
                if not available_spks:
                    raise RuntimeError("No available speakers found in spk2info.pt. Please check your model directory or provide --prompt_wav.")
                default_spk_id = available_spks[0]
                for model_output in cosyvoice.inference_sft(txt, spk_id=default_spk_id, return_token_sequence=True, wm_args=wm_args):
                    wav = torch.tensor(model_output['tts_speech'], dtype=torch.float32, device=device).unsqueeze(0)
                    batch_audio.append(wav)
                    ar_tokens = model_output.get('ar_tokens', None)
                    if ar_tokens is None:
                        raise RuntimeError('CosyVoice model did not return ar_tokens (original token sequence). This is required for watermark detection.')
                    ar_tokens = ar_tokens.squeeze(0) if ar_tokens.dim() == 2 and ar_tokens.shape[0] == 1 else ar_tokens
                    batch_tokens.append(ar_tokens)
                    break
        batch_audio = torch.stack(batch_audio, dim=0)
        for validation_aug, strengths in augs:
            for strength in strengths:
                batch_aug_audio, _ = validation_aug(batch_audio.clone(), None, strength)
                for i in range(batch_aug_audio.shape[0]):
                    synced_audio = batch_aug_audio[i:i+1]
                    # Extract roundtrip semantic tokens using CosyVoiceTokenizerWrapper
                    try:
                        tokens_rt = tokenizer.encode_semantic(synced_audio.squeeze(0))
                    except Exception:
                        tokens_rt = None
                    orig_tokens = batch_tokens[i]
                    # Compute watermark stats for orig tokens if available
                    if orig_tokens is not None:
                        ngrams_orig = build_stream_ngrams_from_full_stream(orig_tokens, args.wm_ngram, device='cpu')
                        s_map = clustering_maps.get(0) if clustering_maps else None
                        g_mask_o, s_mask_o = compute_watermark_scores(
                            orig_tokens, ngrams_orig, semantic_vocab_size, args.wm_gamma, args.wm_seed, clustering_map=s_map
                        )
                        greens_o = float((g_mask_o * s_mask_o).float().sum().item())
                        scored_o = float(s_mask_o.float().sum().item())
                        pval_o = get_binomial_pval(greens_o, scored_o, args.wm_gamma)
                    else:
                        greens_o = scored_o = pval_o = None
                    # Compute watermark stats for rt tokens if available
                    if tokens_rt is not None:
                        ngrams_rt = build_stream_ngrams_from_full_stream(tokens_rt, args.wm_ngram, device='cpu')
                        s_map = clustering_maps.get(0) if clustering_maps else None
                        g_mask_r, s_mask_r = compute_watermark_scores(
                            tokens_rt, ngrams_rt, semantic_vocab_size, args.wm_gamma, args.wm_seed, clustering_map=s_map
                        )
                        greens_r = float((g_mask_r * s_mask_r).float().sum().item())
                        scored_r = float(s_mask_r.float().sum().item())
                        pval_r = get_binomial_pval(greens_r, scored_r, args.wm_gamma)
                    else:
                        greens_r = scored_r = pval_r = None
                    global_results.append({
                        "config": config_name,
                        "idx": batch_start + i,
                        "prompt": batch_texts[i],
                        "aug_name": str(validation_aug),
                        "strength": strength,
                        "greens_orig": greens_o,
                        "scored_orig": scored_o,
                        "pval_orig": pval_o,
                        "greens_rt": greens_r,
                        "scored_rt": scored_r,
                        "pval_rt": pval_r,
                        # Add more fields as needed
                    })
                    if args.save_audio > 0 and (batch_start + i) < args.save_audio:
                        audio_output_dir = os.path.join(args.output_dir, f"audio_{config_name}")
                        os.makedirs(audio_output_dir, exist_ok=True)
                        audio = batch_aug_audio[i, 0].detach().cpu()
                        # Resample from 24kHz to 16kHz if needed (CosyVoice3 output is 24kHz)
                        sr_out = getattr(tokenizer, 'sr_out', None)
                        sr_in = getattr(tokenizer, 'sr', None)
                        if sr_out is not None and sr_in is not None and sr_out != sr_in:
                            import torchaudio
                            audio = torchaudio.functional.resample(audio, sr_out, sr_in)
                        sphn.write_wav(
                            os.path.join(audio_output_dir, f'{validation_aug}_{strength}_{batch_start + i:03d}.wav'),
                            audio.numpy().astype(np.float32),
                            sr,
                        )
        with open(generated_text_path, "a", encoding="utf-8") as f:
            for idx, txt in enumerate(batch_texts):
                f.write(f"{batch_start + idx:04d},{txt}\n")
    torch.save({'config': vars(args), 'results': global_results}, os.path.join(args.output_dir, f'summary_{config_name}.pt'))

    # DataFrame and CSV output (match other unified_wm_*.py scripts)
    df = pd.DataFrame([{
        "config": r.get("config", config_name),
        "idx": r["idx"],
        "prompt": r.get("prompt", None),
        "aug_name": r["aug_name"],
        "strength": str(r["strength"]),
        "greens_orig": r["greens_orig"],
        "scored_orig": r["scored_orig"],
        "pval_orig": r["pval_orig"],
        "greens_rt": r["greens_rt"],
        "scored_rt": r["scored_rt"],
        "pval_rt": r["pval_rt"],
        "logp_orig": -np.log10(r["pval_orig"]) if r["pval_orig"] is not None and r["pval_orig"] > 0 else None,
        "logp_rt": -np.log10(r["pval_rt"]) if r["pval_rt"] is not None and r["pval_rt"] > 0 else None,
    } for r in global_results])

    cols = [c for c in ["greens_orig","scored_orig","greens_rt","scored_rt","pval_orig","pval_rt","logp_orig","logp_rt"] if c in df.columns]
    mean_df = df.groupby(["aug_name", "strength"])[cols].agg("mean")
    mean_df.to_csv(os.path.join(args.output_dir, f"summary_{config_name}.csv"))

    df.to_csv(os.path.join(args.output_dir, f"results_{config_name}.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="CosyVoice Watermark Evaluation Pipeline")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.device_count() else "cpu")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to txt file with prompts")
    parser.add_argument("--prompt_wav", type=str, default=None, help="Path to reference speaker wav file (optional)")
    parser.add_argument("--prompt_text", type=str, default=None, help="Transcript of prompt_wav for zero-shot mode; improves intelligibility and WER")
    parser.add_argument("--nsamples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--wm_method", type=str, default="maryland")
    parser.add_argument("--wm_streams", nargs='+', default=[0], help="CosyVoice uses stream 0")
    parser.add_argument("--wm_delta", type=float, default=2.0)
    parser.add_argument("--wm_gamma", type=float, default=0.25)
    parser.add_argument("--wm_ngram", type=int, default=0)
    parser.add_argument("--wm_seed", type=int, default=0)
    parser.add_argument("--encodec_weight", type=str, default=None)
    parser.add_argument("--save_audio", type=int, default=10)
    parser.add_argument("--eval_aug", type=bool_inst, default=False)
    parser.add_argument("--run_mode", type=str, choices=["base", "general", "ablate", "select"], default="base")
    parser.add_argument("--clustering_dir", type=str, default="models/embeddings/clusterings")
    parser.add_argument("--clustering_pkl", type=str, default="models/embeddings/clusterings_cosyvoice/leiden_clusterings_trainonly_allparams.pkl")
    parser.add_argument("--cosyvoice_model_dir", type=str, default=None,
                        help="Path to a local CosyVoice3 checkpoint directory.")
    # Speaker ID argument removed; default speaker will be used
    parser.add_argument("--cosyvoice_max_tokens", type=int, default=None, help="Max tokens for CosyVoice generation (rarely used; prefer steps or default behavior)")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    seed_all(args.seed)

    if not args.cosyvoice_model_dir:
        parser.error("--cosyvoice_model_dir is required for CosyVoice runs.")

    if args.prompt_wav and not args.prompt_text:
        print(
            "[WARN] --prompt_wav is set without --prompt_text. Falling back to cross-lingual mode, "
            "which can significantly increase WER for same-language cloning. Provide --prompt_text for zero-shot mode.",
            flush=True,
        )

    if args.run_mode == "base":
        configs_to_run = [{"method": None, "maps": None, "name": "base"}]
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
