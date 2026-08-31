#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
import struct
import wave

from demo_common import (
    DEMO_MODES,
    ROOT,
    choose_python_bin,
    cleanup_demo_tmp,
    ensure_prompt_file,
    iter_demo_modes,
    print_summary,
    require_existing_dir,
    resolve_summary_path,
    run_eval_command,
    summarize_results,
)


REQUIRED_MODULES = (
    "torch",
    "torchaudio",
    "onnxruntime",
    "whisper",
    "hydra",
    "lightning",
    "diffusers",
    "conformer",
    "inflect",
    "rich",
    "x_transformers",
)


def ensure_prompt_audio(prompt_audio: str = None):
    if prompt_audio:
        return prompt_audio

    prompt_dir = ROOT / "tmp" / "demo_cosyvoice"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / "prompt.wav"

    sr = 16000
    secs = 0.2
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

    return str(Path("tmp") / "demo_cosyvoice" / "prompt.wav")


def run_eval(args, mode: str):
    clustered = mode == "clustered"
    output_dir = Path(args.output_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    python_bin = choose_python_bin(args.python_bin, REQUIRED_MODULES)

    eval_command = [
        python_bin,
        "-m",
        "evals.wm_cosyvoice",
        "--output_dir",
        str(output_dir),
        "--prompt_file",
        args.prompt_file,
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
        "0",
        "--wm_delta",
        str(args.wm_delta),
        "--wm_ngram",
        str(args.wm_ngram),
        "--save_audio",
        "3",
        "--eval_aug",
        "false",
        "--run_mode",
        "select" if clustered else "base",
        "--device",
        args.device,
        "--cosyvoice_model_dir",
        args.cosyvoice_model_dir,
    ]
    if args.prompt_wav:
        eval_command.extend(["--prompt_wav", args.prompt_wav])
    if args.prompt_text:
        eval_command.extend(["--prompt_text", args.prompt_text])
    if args.cosyvoice_max_tokens is not None:
        eval_command.extend(["--cosyvoice_max_tokens", str(args.cosyvoice_max_tokens)])

    run_eval_command(eval_command)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Small CosyVoice demo for the nograd-audio-wm artifact.")
    parser.add_argument("--prompt_file", default=None, help="Text file with one prompt per line. If omitted, the script creates a tiny demo prompt file.")
    parser.add_argument("--output_dir", default="outputs/demo_cosyvoice")
    parser.add_argument("--nsamples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=0)
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
    parser.add_argument("--cosyvoice_model_dir", default="checkpoints/cosyvoice/Fun-CosyVoice3-0.5B")
    parser.add_argument("--prompt_wav", default=None)
    parser.add_argument("--prompt_text", default=None)
    parser.add_argument("--cosyvoice_max_tokens", type=int, default=None)
    parser.add_argument("--python-bin", default=None, help="Python interpreter to use for the actual evaluation run.")
    args = parser.parse_args()

    args.prompt_file = ensure_prompt_file(
        args.prompt_file,
        subdir="demo_cosyvoice",
        filename="prompts.txt",
        lines=[
            "Please describe the weather in one short sentence.",
            "Tell me about your favorite book in one sentence.",
            "Explain what you had for breakfast today.",
        ],
    )
    args.cosyvoice_model_dir = require_existing_dir(args.cosyvoice_model_dir, "--cosyvoice_model_dir")
    args.prompt_wav = ensure_prompt_audio(args.prompt_wav)
    if args.prompt_wav:
        args.prompt_wav = str(Path(args.prompt_wav))
    if args.prompt_text is None:
        args.prompt_text = "This is a short prompt."
    for mode in iter_demo_modes(args.mode):
        output_dir = run_eval(args, mode)
        if mode == "clustered":
            summary_path = resolve_summary_path(output_dir, clustered=True)
        else:
            summary_path = output_dir / "summary_base.pt"
        summary = summarize_results(summary_path, orig_key="pval_orig", rt_key="pval_rt")
        print_summary(mode, summary, row_rt_key="pval_rt")
    
    cleanup_demo_tmp("demo_cosyvoice")


if __name__ == "__main__":
    main()