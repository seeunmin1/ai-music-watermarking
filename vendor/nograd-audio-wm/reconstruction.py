import argparse
import os
import glob
import csv
from typing import List
import torch
import numpy as np
import torchaudio
from huggingface_hub import hf_hub_download
from collections import defaultdict

from models.moshi.models.loaders import get_mimi
from models.moshi.models import loaders
from transformers import EncodecModel, AutoProcessor


# Add this helper function
def load_encodec(weight_path, device):
    model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(device)
    if weight_path:
        raw_sd = torch.load(weight_path, map_location=device)
        if "model_state" in raw_sd: raw_sd = raw_sd["model_state"]
        new_sd = {}
        for k, v in raw_sd.items():
            nk = k.replace("encoder.model", "encoder.layers").replace("decoder.model", "decoder.layers").replace("quantizer.vq", "quantizer.layers").replace("conv.conv.", "conv.")
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model



def list_audio_files(audio_dir: str, exts=("wav", "mp3", "ogg", "flac")) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(audio_dir, f"**/*.{ext}"), recursive=True))
    return sorted(files)


def resample_if_needed(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return waveform
    return torchaudio.transforms.Resample(src_sr, dst_sr)(waveform)


def single_channel_best_match(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.shape[0] == b.shape[0]:
        return float((a == b).mean())
    if a.shape[0] < b.shape[0]:
        a, b = b, a
    best = 0.0
    for shift in range(a.shape[0]):
        rolled = np.roll(a, shift)[: b.shape[0]]
        best = max(best, (rolled == b).mean())
    return float(best)


def per_channel_token_match(toks1: torch.LongTensor, toks2: torch.LongTensor):
    if toks1.dim() == 3:
        toks1 = toks1.squeeze(0)
    if toks2.dim() == 3:
        toks2 = toks2.squeeze(0)
    return [
        single_channel_best_match(
            toks1[ch].cpu().numpy().astype(np.int64),
            toks2[ch].cpu().numpy().astype(np.int64),
        )
        for ch in range(toks1.shape[0])
    ]


def save_tokens_txt(path: str, codes: torch.LongTensor):
    if codes.dim() == 3:
        codes = codes.squeeze(0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ch in range(codes.shape[0]):
            f.write(" ".join(map(str, codes[ch].tolist())) + "\n")


def run_pipeline(
    weight_path: str,
    audio_dir: str,
    output_dir: str,
    nsamples: int = 0,
    batch_size: int = 1,
    is_encodec: bool = False, # Added flag
):
    per_channel_matches = defaultdict(list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    if is_encodec:
        print("Loading EnCodec...")
        model = load_encodec(weight_path, device)
        sr = 32000
    else:
        print("Loading MIMI...")
        model = get_mimi(weight_path, device=device)
        sr = model.sample_rate

    audio_files = list_audio_files(audio_dir)
    if nsamples > 0:
        audio_files = audio_files[:nsamples]

    csv_rows = []

    print("Found", len(audio_files), "files.")

    for batch_start in range(0, len(audio_files), batch_size):
        batch_files = audio_files[batch_start: batch_start + batch_size]
        batch_wavs = []

        for p in batch_files:
            wav, wsr = torchaudio.load(p)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = resample_if_needed(wav, wsr, sr)
            batch_wavs.append(wav)

        if not batch_wavs:
            continue

        max_len = max(w.shape[-1] for w in batch_wavs)
        batch_wavs = [
            torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
            for w in batch_wavs
        ]
        batch_wavs = torch.stack(batch_wavs, dim=0).to(device)  # [B, 1, T]

        with torch.no_grad():
            if is_encodec:
                # Encodec API returns batch in .audio_codes
                orig_tokens = model.encode(batch_wavs).audio_codes.squeeze(0) # [B, K, T]
                decoded_audio = model.decode(orig_tokens.unsqueeze(0), [None]*batch_wavs.shape[0]).audio_values
                rt_tokens = model.encode(decoded_audio).audio_codes.squeeze(0)
            else:
                orig_tokens = model.encode(batch_wavs)
                decoded_audio = model.decode(orig_tokens)
                rt_tokens = model.encode(decoded_audio)

        for i, audio_path in enumerate(batch_files):
            idx = batch_start + i

            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_dir = os.path.join(output_dir, "samples", base)
            os.makedirs(out_dir, exist_ok=True)

            orig = orig_tokens[i]
            rt = rt_tokens[i]

            save_tokens_txt(os.path.join(out_dir, "orig_tokens.txt"), orig)
            save_tokens_txt(os.path.join(out_dir, "roundtrip_tokens.txt"), rt)

            rates = per_channel_token_match(orig, rt)
            mean_match = float(np.mean(rates)) if rates else 0.0

            for ch, rate in enumerate(rates):
                per_channel_matches[ch].append(rate)

            csv_rows.append({
                "idx": idx,
                "audio_file": os.path.basename(audio_path),
                "orig_len": orig.shape[-1],
                "rt_len": rt.shape[-1],
                "mean_match": mean_match,
                "rcc_error": 1.0 - mean_match,
                "per_channel": rates,
            })

            print(f"[{idx}] {os.path.basename(audio_path)} mean_match={mean_match:.4f}")

    max_k = max((len(r["per_channel"]) for r in csv_rows), default=0)
    fieldnames = ["idx", "audio_file", "orig_len", "rt_len", "mean_match", "rcc_error"] + \
                 [f"tm_rate_{i}" for i in range(max_k)]

    csv_path = os.path.join(output_dir, "rcc_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            row = {
                "idx": r["idx"],
                "audio_file": r["audio_file"],
                "orig_len": r["orig_len"],
                "rt_len": r["rt_len"],
                "mean_match": f"{r['mean_match']:.4f}",
                "rcc_error": f"{r['rcc_error']:.4f}",
            }
            for i in range(max_k):
                row[f"tm_rate_{i}"] = (
                    f"{r['per_channel'][i]:.4f}" if i < len(r["per_channel"]) else ""
                )
            writer.writerow(row)

    print(f"Done. Summary CSV: {csv_path}")

    avg_match_path = os.path.join(output_dir, "avg_per_channel_match.txt")
    with open(avg_match_path, "w") as f:
        for ch in sorted(per_channel_matches.keys()):
            errs = per_channel_matches[ch]
            if len(errs) == 0:
                continue
            mean_err = float(np.mean(errs))
            f.write(f"channel_{ch}: {mean_err:.6f}\n")

    print(f"Per-channel matches: {avg_match_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weight_path", default=None) # Renamed from mimi_weight_ori
    parser.add_argument("--nsamples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--encodec", action="store_true", help="Use EnCodec instead of Mimi")
    args = parser.parse_args()

    if args.weight_path is None and not args.encodec:
        args.weight_path = hf_hub_download(
            "kyutai/moshiko-pytorch-bf16",
            loaders.MIMI_NAME,
        )

    run_pipeline(
        weight_path=args.weight_path,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        is_encodec=args.encodec
    )
