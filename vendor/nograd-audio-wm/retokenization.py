#!/usr/bin/env python3
"""
  1) Samples the Moshi LM using existing audio prompts (same logic as the repo's evals)
  2) Saves generated waveforms and the generated token sequences as files (tokens as .txt)
  3) Retokenizes generated audio with MIMI (in-memory)
  4) Computes RCC / token-match stats and saves a CSV summary (does NOT include saved file paths)

Notes:
 - Generated wavs are saved under output_dir/sample_{idx}/generated.wav
 - Original generated tokens are saved as output_dir/sample_{idx}/orig_generated_tokens.txt
   (one line per channel, space-separated token ids)
 - Roundtrip tokens are saved as output_dir/sample_{idx}/roundtrip_tokens.txt
 - The CSV includes: idx, prompt_file, orig_len, rt_len, mean_match, rcc_error, tm_rate_0..tm_rate_{K-1}
 - Match rates in CSV are written with 4 decimal places.
"""

import argparse
import os
import glob
import csv
from typing import List
import torch
import numpy as np
import torchaudio
from huggingface_hub import hf_hub_download

from models.moshi.models import loaders
from models.moshi.models.loaders import get_mimi, get_moshi_lm
from models.moshi.models.lm import LMGen


def list_audio_files(audio_dir: str, exts=("wav", "mp3", "ogg", "flac")) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(audio_dir, f"*.{ext}"), recursive=False))
    files = sorted(files)
    return files


def resample_if_needed(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return waveform
    # waveform shape [C, T] or [1, T]
    resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)
    return resampler(waveform)


def single_channel_best_match(a: np.ndarray, b: np.ndarray) -> float:
    """
    Best circular alignment match ratio between 1D integer arrays a and b.
    Mirrors the rolling-match logic used in token_match.py
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.shape[0] == b.shape[0]:
        return float((a == b).mean())
    if a.shape[0] < b.shape[0]:
        a, b = b, a
    L_long, L_short = a.shape[0], b.shape[0]
    best = 0.0
    # Try shifts (circular roll) of longer sequence
    for shift in range(L_long):
        rolled = np.roll(a, shift)[:L_short]
        match = (rolled == b).mean()
        if match > best:
            best = match
    return float(best)


def per_channel_token_match(toks1: torch.LongTensor, toks2: torch.LongTensor) -> List[float]:
    """
    toks1 and toks2 : torch tensors with shape [K, T1] and [K, T2] (or [1, K, T])
    Returns per-channel best-match rates (list of K floats)
    """
    if toks1.dim() == 3:
        toks1 = toks1.squeeze(0)
    if toks2.dim() == 3:
        toks2 = toks2.squeeze(0)
    K = toks1.shape[0]
    rates = []
    for ch in range(K):
        a = toks1[ch].cpu().numpy().astype(np.int64)
        b = toks2[ch].cpu().numpy().astype(np.int64)
        rates.append(single_channel_best_match(a, b))
    return rates


def save_tokens_txt(path: str, codes: torch.LongTensor):
    """
    Save codes tensor to a text file: one line per channel, space-separated token ids.
    codes: tensor shape [K, T] or [1, K, T] or [K, T] on CPU
    """
    if codes.dim() == 3:
        codes = codes.squeeze(0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ch in range(codes.shape[0]):
            row = " ".join(str(int(x)) for x in codes[ch].tolist())
            f.write(row + "\n")


def run_pipeline(
    mimi_weight_ori: str,
    moshi_weight: str,
    audio_dir: str,
    output_dir: str,
    nsamples: int = 50,
    batch_size: int = 1,
    steps: int = 200
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    print("Loading models...", flush=True)

    # Load tokenizer / autoencoder (MIMI) and original tokenizer for encoding prompts
    mimi_ori = get_mimi(mimi_weight_ori, device=device)
    # Load LM and wrapper
    lm = get_moshi_lm(moshi_weight, device=device)
    lm_gen = LMGen(lm)
    lm_gen.temp = 0.8
    # disable watermarking for sampling unwatermarked audio
    lm_gen.wm = "none"
    lm_gen.wm_streams = []

    frame_size = int(mimi_ori.sample_rate / mimi_ori.frame_rate)
    print(f"MIMI sample_rate={mimi_ori.sample_rate}, frame_rate={mimi_ori.frame_rate}, frame_size={frame_size}")

    audio_files = list_audio_files(audio_dir)
    if nsamples > 0:
        audio_files = audio_files[:nsamples]
    print(f"Using {len(audio_files)} prompt files from {audio_dir}")

    csv_rows = []

    for batch_start in range(0, len(audio_files), batch_size):
        batch_files = audio_files[batch_start: batch_start + batch_size]
        batch_pcms = []
        # Load and resample prompts
        for p in batch_files:
            wav, sr = torchaudio.load(p)  # shape [C, T]
            # convert to mono if necessary
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            wav = resample_if_needed(wav, sr, mimi_ori.sample_rate)
            # ensure shape [1, T] -> add batch dim later
            batch_pcms.append(wav)

        if len(batch_pcms) == 0:
            continue

        # Pad batch to same length and add batch dim
        max_len = max(w.shape[-1] for w in batch_pcms)
        batch_pcms = [torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) for w in batch_pcms]
        batch_pcms = torch.stack(batch_pcms, dim=0)  # [B, C, T] where C=1

        with torch.no_grad():
            # Encode prompts with mimi_ori (used as prompt encoder in the repo)
            # prompt_codes shape: [B, K, T_codes]
            prompt_codes = mimi_ori.encode(batch_pcms.to(device))

            # compute start_answer_idx like evals/main_wm.py: wait until prompt codes consumed before collecting generated tokens
            start_answer_idx = prompt_codes.shape[-1] if prompt_codes is not None else 0

            # Generation loop (streaming): feed codes frame-by-frame like evals/main_wm.py
            batch_all_tokens = []
            batch_all_audios = []

            # PASS batch size to streaming()
            batch_B = batch_pcms.shape[0]
            with lm_gen.streaming(batch_B):
                for step in range(steps):
                    if prompt_codes is not None and step < prompt_codes.shape[-1]:
                        codes = prompt_codes[:, :, step: step+1]  # [B, K, 1]
                    else:
                        chunk = torch.zeros((batch_B, 1, frame_size), dtype=torch.float32, device=device)
                        codes = mimi_ori.encode(chunk)  # [B, K, 1]
                    # set force_epad at the boundary where generation answer should begin (mirrors evals)
                    tokens = lm_gen.step(codes[:, :, :1], force_epad=(step == start_answer_idx))
                    if tokens is None:
                        continue
                    # skip collecting tokens that correspond to prompt processing (only collect generated tokens)
                    if prompt_codes is not None and step < start_answer_idx:
                        continue

                    # Collect tokens and decoded audio chunk
                    batch_all_tokens.append(tokens.detach().cpu())  # tokens on CPU
                    audio_tokens = tokens[:, 1:, :]  # skip text stream, keep audio codebooks
                    pcms = mimi_ori.decode(audio_tokens.to(device))  # [B, 1, frame_samples]
                    batch_all_audios.append(pcms.detach().cpu())

        if len(batch_all_audios) == 0 or len(batch_all_tokens) == 0:
            print("No audio/tokens generated for this batch; skipping.")
            continue

        # Concatenate along time
        batch_all_audio = torch.cat(batch_all_audios, dim=-1)  # [B, 1, S]
        batch_all_tokens_th = torch.cat(batch_all_tokens, dim=-1)  # [B, 1+K, S_frames]

        # For each sample in batch compute roundtrip and stats (save wav + tokens to files, but don't include paths in CSV)
        for idx_in_batch, audio_path in enumerate(batch_files):
            global_idx = batch_start + idx_in_batch
            out_prefix = os.path.join(output_dir, f"sample_{global_idx:05d}")
            os.makedirs(out_prefix, exist_ok=True)

            gen_tokens = batch_all_tokens_th[idx_in_batch]  # [1+K, T_orig]
            # Save generated wav
            wav_tensor = batch_all_audio[idx_in_batch]  # [1, S]
            wav_to_save = wav_tensor.cpu().float()  # ensure cpu & float32
            wav_out_path = os.path.join(out_prefix, "generated.wav")
            # torchaudio.save expects shape [channels, time]
            torchaudio.save(wav_out_path, wav_to_save, sample_rate=mimi_ori.sample_rate)

            # Save original generated tokens as txt (one line per channel, skip text stream)
            if gen_tokens.shape[0] > 1:
                orig_audio_only = gen_tokens[1:, :]  # [K, T_orig]
                orig_tokens_txt = os.path.join(out_prefix, "orig_generated_tokens.txt")
                save_tokens_txt(orig_tokens_txt, orig_audio_only)
            else:
                # write empty orig tokens file for consistency
                orig_tokens_txt = os.path.join(out_prefix, "orig_generated_tokens.txt")
                with open(orig_tokens_txt, "w") as f:
                    f.write("")

            # Re-encode the in-memory waveform and save roundtrip tokens as txt
            with torch.no_grad():
                wav_for_rt = wav_tensor.unsqueeze(0) if wav_tensor.dim() == 2 else wav_tensor  # ensure [1,1,S]
                # some shapes might be [1,S] or [1,1,S]; ensure shape [1,1,S] -> mimi.encode expects [B, C, T]
                if wav_for_rt.dim() == 2:
                    wav_for_rt = wav_for_rt.unsqueeze(1)
                rt_tokens = mimi_ori.encode(wav_for_rt.to(device)).detach().cpu()  # [1, K, T_rt]
            # save rt tokens text (one line per channel)
            if rt_tokens.numel() > 0:
                rt_audio_only = rt_tokens[0]
                rt_tokens_txt = os.path.join(out_prefix, "roundtrip_tokens.txt")
                save_tokens_txt(rt_tokens_txt, rt_audio_only)
            else:
                rt_tokens_txt = os.path.join(out_prefix, "roundtrip_tokens.txt")
                with open(rt_tokens_txt, "w") as f:
                    f.write("")

            # Compute RCC / token-match stats (do not include file paths in CSV)
            if gen_tokens.shape[0] <= 1:
                # no audio codebooks present
                per_ch_rates = []
                mean_match = 0.0
                orig_len = 0
                rt_len = 0
            else:
                rt_audio_only = rt_tokens[0]        # [K, T_rt]
                per_ch_rates = per_channel_token_match(orig_audio_only, rt_audio_only)
                mean_match = float(np.mean(per_ch_rates)) if per_ch_rates else 0.0
                # sequence lengths are the same across channels, so save single values
                orig_len = int(orig_audio_only.shape[-1])
                rt_len = int(rt_audio_only.shape[-1])

            rcc_error = 1.0 - mean_match

            # store minimal info (prompt filename, lengths, match rates)
            row = {
                "idx": global_idx,
                "prompt_file": os.path.basename(audio_path),
                "orig_len": orig_len,
                "rt_len": rt_len,
                "mean_match": mean_match,
                "rcc_error": rcc_error,
                "per_channel": per_ch_rates,
            }
            csv_rows.append(row)

            print(f"[{global_idx}] prompt={os.path.basename(audio_path)} mean_match={mean_match:.4f} rcc_error={rcc_error:.4f}")

    # Write summary CSV with per-channel columns (tm_rate_i) and the single sequence lengths
    summary_csv = os.path.join(output_dir, "rcc_summary.csv")
    # find maximum number of channels present across samples
    max_k = 0
    for r in csv_rows:
        k = len(r.get("per_channel", []))
        if k > max_k:
            max_k = k

    # prepare header: idx, prompt_file, orig_len, rt_len, mean_match, rcc_error, tm_rate_0..tm_rate_{max_k-1}
    base_fields = ["idx", "prompt_file", "orig_len", "rt_len", "mean_match", "rcc_error"]
    tm_fields = [f"tm_rate_{i}" for i in range(max_k)]
    fieldnames = base_fields + tm_fields

    with open(summary_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            # format mean_match and rcc_error to 4 decimal places
            out_row = {
                "idx": r.get("idx", ""),
                "prompt_file": r.get("prompt_file", ""),
                "orig_len": r.get("orig_len", 0),
                "rt_len": r.get("rt_len", 0),
                "mean_match": f"{r.get('mean_match', 0.0):.4f}",
                "rcc_error": f"{r.get('rcc_error', 0.0):.4f}",
            }
            per_ch = r.get("per_channel", [])
            for i in range(max_k):
                if i < len(per_ch):
                    out_row[f"tm_rate_{i}"] = f"{per_ch[i]:.4f}"
                else:
                    out_row[f"tm_rate_{i}"] = ""
            writer.writerow(out_row)

    print(f"Pipeline finished. Summary CSV: {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mimi_weight_ori", type=str, default=None)
    parser.add_argument("--moshi_weight", type=str, default=None)
    parser.add_argument("--nsamples", type=int, default=50, help="Number of prompt files to process (0 = all)")
    parser.add_argument("--steps", type=int, default=200, help="Number of generation steps (frames)")
    args = parser.parse_args()

    # Download weights if not provided
    if args.mimi_weight_ori is None:
        args.mimi_weight_ori = hf_hub_download("kyutai/moshiko-pytorch-bf16", loaders.MIMI_NAME)
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download("kyutai/moshiko-pytorch-bf16", loaders.MOSHI_NAME)

    run_pipeline(
        mimi_weight_ori=args.mimi_weight_ori,
        moshi_weight=args.moshi_weight,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        nsamples=args.nsamples,
        steps=args.steps
    )
