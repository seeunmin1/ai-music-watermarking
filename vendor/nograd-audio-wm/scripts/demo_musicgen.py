#!/usr/bin/env python3

import argparse
from pathlib import Path

from demo_common import (
    DEMO_MODES,
    ROOT,
    choose_python_bin,
    ensure_prompt_file,
    iter_demo_modes,
    print_summary,
    resolve_summary_path,
    run_eval_command,
    summarize_results,
)


REQUIRED_MODULES = ("torch", "torchaudio", "transformers", "huggingface_hub")
DEFAULT_REPO_PROMPT_FILE = ROOT / "data" / "music_prompts.txt"


def run_eval(args, mode: str):
    clustered = mode == "clustered"
    output_dir = Path(args.output_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    python_bin = choose_python_bin(args.python_bin, REQUIRED_MODULES)

    eval_command = [
        python_bin,
        "-m",
        "evals.main_wm_music",
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
        "1",
        "2",
        "3",
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
    if args.encodec_weight:
        eval_command.extend(["--encodec_weight", args.encodec_weight])
    if clustered:
        eval_command.extend(["--wm_clustering", "true"])

    run_eval_command(eval_command)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Small MusicGen demo for the nograd-audio-wm artifact.")
    parser.add_argument("--prompt_file", default=None, help="Text file with one music prompt per line. If omitted, the script creates a tiny demo prompt file.")
    parser.add_argument("--output_dir", default="outputs/demo_musicgen")
    parser.add_argument("--nsamples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=256)
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
    parser.add_argument("--encodec_weight", default=None)
    parser.add_argument("--python-bin", default=None, help="Python interpreter to use for the actual evaluation run.")
    args = parser.parse_args()
    if args.prompt_file is None and DEFAULT_REPO_PROMPT_FILE.exists():
        args.prompt_file = str(DEFAULT_REPO_PROMPT_FILE)
    args.prompt_file = ensure_prompt_file(
        args.prompt_file,
        subdir="demo_musicgen",
        filename="prompts.txt",
        lines=[
            "Warm ambient synth with sparse piano.",
            "Upbeat electronic dance music with heavy bass.",
            "Soft acoustic guitar with gentle strings.",
        ],
    )

    for mode in iter_demo_modes(args.mode):
        output_dir = run_eval(args, mode)
        if mode == "clustered":
            summary_path = resolve_summary_path(output_dir, clustered=True)
        else:
            summary_path = output_dir / "summary_base.pt"
        summary = summarize_results(summary_path, orig_key="original_pval", rt_key="pval")
        print_summary(mode, summary, row_rt_key="pval")


if __name__ == "__main__":
    main()