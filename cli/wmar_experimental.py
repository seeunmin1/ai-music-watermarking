"""Experimental adapter boundary for g-milis/nograd-audio-wm.

The upstream repository is vendored under vendor/nograd-audio-wm and is not
executed by default because it requires heavyweight ML dependencies,
checkpoints, and codec assets. This module gives the CLI a stable integration
point for future WMAR clustered-token detection without mixing research code
into the production provenance verdict.
"""

from __future__ import annotations

from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "nograd-audio-wm"
UPSTREAM_URL = "https://github.com/g-milis/nograd-audio-wm"

SUPPORTED_MODEL_FAMILIES = {
    "musicgen_encodec": "evals/main_wm_music.py",
    "moshi_mimi": "evals/main_wm.py",
    "cosyvoice": "evals/wm_cosyvoice.py",
    "sparktts": "evals/wm_sparktts.py",
}

DEFAULT_COMMANDS = {
    "musicgen_encodec": [
        "python", "-m", "evals.main_wm_music",
        "--output_dir", "outputs/musicgen_clustered",
        "--prompt_file", "data/music_prompts.txt",
        "--nsamples", "3",
        "--batch_size", "1",
        "--steps", "256",
        "--wm_method", "maryland",
        "--wm_streams", "0", "1", "2", "3",
        "--wm_delta", "2.0",
        "--wm_ngram", "0",
        "--wm_clustering", "true",
    ],
    "moshi_mimi": [
        "python", "-m", "evals.main_wm",
        "--output_dir", "outputs/moshi_clustered",
        "--audio_dir", "data/audio_prompts",
        "--nsamples", "3",
        "--batch_size", "1",
        "--steps", "200",
        "--wm_method", "maryland",
        "--wm_streams", "1", "2", "3", "4",
        "--wm_delta", "2.0",
        "--wm_ngram", "0",
        "--wm_clustering", "true",
    ],
    "cosyvoice": [
        "python", "-m", "evals.wm_cosyvoice",
        "--output_dir", "outputs/cosyvoice",
        "--prompt_file", "data/tts_prompts.txt",
        "--wm_method", "maryland",
        "--wm_streams", "0",
        "--run_mode", "select",
    ],
    "sparktts": [
        "python", "-m", "evals.wm_sparktts",
        "--output_dir", "outputs/sparktts",
        "--prompt_file", "data/tts_prompts.txt",
        "--wm_method", "maryland",
        "--wm_streams", "0",
        "--run_mode", "select",
    ],
}


def describe_adapter() -> dict:
    available = VENDOR_ROOT.exists()
    return {
        "available": available,
        "vendor_root": str(VENDOR_ROOT),
        "upstream": UPSTREAM_URL,
        "supported_model_families": SUPPORTED_MODEL_FAMILIES,
        "status": "ready_for_experimental_eval" if available else "vendor_missing",
    }


def build_eval_command(model_family: str) -> dict:
    if model_family not in DEFAULT_COMMANDS:
        return {
            "ok": False,
            "error": f"unsupported model family: {model_family}",
            "supported_model_families": sorted(DEFAULT_COMMANDS),
        }
    return {
        "ok": True,
        "cwd": str(VENDOR_ROOT),
        "command": DEFAULT_COMMANDS[model_family],
        "note": "Run after installing vendor/nograd-audio-wm requirements and downloading required checkpoints.",
    }
