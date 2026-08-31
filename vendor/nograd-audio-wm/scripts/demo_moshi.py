#!/usr/bin/env python3

import argparse
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
import struct
import wave
from demo_common import DEMO_MODES, iter_demo_modes, resolve_summary_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_AUDIO_DIR = ROOT / "data" / "audio_prompts"
DEFAULT_SMOKE_AUDIO_DIR = Path("/fs/nexus-scratch/milis/outputs/wmar_audio/beacon_outputs/audio_prompts")
REQUIRED_MODULES = ("torch", "torchaudio", "transformers", "huggingface_hub")


def has_required_modules(python_bin: str):
    command = [
        python_bin,
        "-c",
        (
            "import importlib.util; "
            f"mods={REQUIRED_MODULES!r}; "
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
            "print('|'.join(missing))"
        ),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return False, list(REQUIRED_MODULES)

    missing = [item for item in result.stdout.strip().split("|") if item]
    return len(missing) == 0, missing


def choose_python_bin(requested: str = None):
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend([
        sys.executable,
        os.environ.get("VIRTUAL_ENV") and str(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"),
        "python",
        "python3",
        "/usr/bin/python3",
        "/usr/bin/python3.11",
    ])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ok, missing = has_required_modules(candidate)
        if ok:
            return candidate

    missing_hint = ", ".join(REQUIRED_MODULES)
    raise SystemExit(
        "No Python interpreter with the required runtime packages was found. "
        f"Install the artifact environment first, then rerun the demo. Missing modules: {missing_hint}."
    )


def ensure_prompt_audio(audio_dir: str = None):
    if audio_dir:
        return audio_dir

    if DEFAULT_REPO_AUDIO_DIR.exists():
        return str(DEFAULT_REPO_AUDIO_DIR)

    if DEFAULT_SMOKE_AUDIO_DIR.exists():
        return str(DEFAULT_SMOKE_AUDIO_DIR)

    prompt_dir = ROOT / "tmp" / "demo_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / "prompt_00000.wav"

    sr = 16000
    secs = 0.08
    freq = 220.0
    samples = int(sr * secs)
    with wave.open(str(prompt_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        frames = bytearray()
        for i in range(samples):
            env = min(1.0, i / (0.1 * sr)) * min(1.0, (samples - i) / (0.1 * sr))
            value = int(0.2 * env * 32767 * math.sin(2 * math.pi * freq * i / sr))
            frames.extend(struct.pack("<h", value))
        wav_file.writeframes(frames)

    return str(Path("tmp") / "demo_prompts")


def run_eval(args, mode: str):
    clustered = mode == "clustered"
    output_dir = Path(args.output_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    python_bin = choose_python_bin(args.python_bin)

    eval_command = [
        python_bin,
        "-m",
        "evals.main_wm",
        "--output_dir",
        str(output_dir),
        "--audio_dir",
        args.audio_dir,
        "--nsamples",
        str(args.nsamples),
        "--batch_size",
        "1",
        "--temperature",
        str(args.temperature),
        "--steps",
        str(args.steps),
        "--wm_method",
        "maryland",
        "--wm_streams",
        "1",
        "2",
        "3",
        "4",
        "--wm_delta",
        str(args.wm_delta),
        "--wm_ngram",
        str(args.wm_ngram),
        "--save_audio",
        "3",
        "--eval_aug",
        "false",
        "--device",
        args.device,
    ]

    if args.mimi_weight:
        eval_command.extend(["--mimi_weight", args.mimi_weight])
    if args.mimi_weight_ori:
        eval_command.extend(["--mimi_weight_ori", args.mimi_weight_ori])
    if args.moshi_weight:
        eval_command.extend(["--moshi_weight", args.moshi_weight])
    if args.tokenizer:
        eval_command.extend(["--tokenizer", args.tokenizer])
    if clustered:
        eval_command.extend(["--wm_clustering", "true"])

    command = list(eval_command)
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    return output_dir


def summarize(summary_path: Path):
    import torch

    summary = torch.load(summary_path, map_location="cpu")
    results = summary.get("results", [])
    if not results:
        return None

    orig_pvals = [row["original_pval"] for row in results if row.get("original_pval") is not None]
    rt_pvals = [row["pval"] for row in results if row.get("pval") is not None]

    def median(values):
        if not values:
            return None
        s = sorted(values)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    def mean_log10(values):
        safe = [max(value, 1e-300) for value in values]
        return sum(-math.log10(value) for value in safe) / len(safe) if safe else None

    return {
        "n": len(results),
        "orig_median_pval": median(orig_pvals),
        "orig_mean_logp": mean_log10(orig_pvals),
        "rt_median_pval": median(rt_pvals),
        "rt_mean_logp": mean_log10(rt_pvals),
        "rows": results,
    }


def print_summary(label: str, data):
    if data is None:
        print(f"{label}: no results")
        return

    print(f"\n[{label}]")
    print(f"samples: {data['n']}")
    print(f"median original p-value: {data['orig_median_pval']:.4e}")
    print(f"mean original -log10 p-value: {data['orig_mean_logp']:.4f}")
    if data["rt_median_pval"] is not None:
        print(f"median round-trip p-value: {data['rt_median_pval']:.4e}")
        print(f"mean round-trip -log10 p-value: {data['rt_mean_logp']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Small conversational-audio demo for the nograd-audio-wm artifact.")
    parser.add_argument("--audio_dir", default=None, help="Directory containing prompt audio files. If omitted, the script uses the local smoke-test prompt directory when available, otherwise it creates a tiny synthetic prompt.")
    parser.add_argument("--output_dir", default="outputs/demo_moshi")
    parser.add_argument("--nsamples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--wm_delta", type=float, default=2.0)
    parser.add_argument("--wm_ngram", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode",
        choices=DEMO_MODES,
        default="both",
        help="Run the KGW base watermark, the clustered WMAR method, or both for comparison.",
    )
    parser.add_argument("--mimi_weight", default=None)
    parser.add_argument("--mimi_weight_ori", default=None)
    parser.add_argument("--moshi_weight", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--python-bin", default=None, help="Python interpreter to use for the actual evaluation run.")
    args = parser.parse_args()
    args.audio_dir = ensure_prompt_audio(args.audio_dir)
    for mode in iter_demo_modes(args.mode):
        output_dir = run_eval(args, mode)
        if mode == "clustered":
            summary_path = resolve_summary_path(output_dir, clustered=True)
        else:
            summary_path = output_dir / "summary_base.pt"
        summary = summarize(summary_path)
        print_summary(mode, summary)


if __name__ == "__main__":
    main()