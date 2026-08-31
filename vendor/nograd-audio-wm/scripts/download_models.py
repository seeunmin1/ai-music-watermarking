#!/usr/bin/env python3

import argparse
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]

DELTA_URLS = {
    "mimi_ft_noaug_delta.pth": "https://dl.fbaipublicfiles.com/wmar/finetunes/mimi_ft_noaug_delta.pth",
    "mimi_ft_delta.pth": "https://dl.fbaipublicfiles.com/wmar/finetunes/mimi_ft_delta.pth",
}

TARGETS = {
    "moshi": ("kyutai/moshiko-pytorch-bf16", "checkpoints/moshi"),
    "musicgen": ("facebook/musicgen-medium", "checkpoints/musicgen-medium"),
    "encodec": ("facebook/encodec_32khz", "checkpoints/encodec_32khz"),
    "cosyvoice": ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "checkpoints/cosyvoice/Fun-CosyVoice3-0.5B"),
    "sparktts": ("SparkAudio/Spark-TTS-0.5B", "checkpoints/sparktts/Spark-TTS-0.5B"),
}


def download_repo(repo_id: str, local_dir: Path):
    from huggingface_hub import snapshot_download

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))


def download_deltas(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in DELTA_URLS.items():
        destination = output_dir / filename
        if destination.exists():
            print(f"Already exists: {destination}")
            continue
        print(f"Downloading {url} -> {destination}")
        urlretrieve(url, destination)


def main():
    parser = argparse.ArgumentParser(description="Download public checkpoints used by the nograd-audio-wm artifact.")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGETS.keys()) + ["deltas"],
        default=None,
        help="Subset of assets to download. Defaults to all public assets.",
    )
    parser.add_argument("--all", action="store_true", help="Download all public assets.")
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        default=None,
        help=(
            "Root directory for downloaded checkpoints. "
            "Defaults to <repo_root>/checkpoints. "
            "Useful when placing large model files on a separate volume or scratch space."
        ),
    )
    args = parser.parse_args()

    checkpoints_root = args.checkpoints_dir if args.checkpoints_dir else ROOT / "checkpoints"

    selected_targets = args.targets or []
    if args.all or not selected_targets:
        selected_targets = ["moshi", "musicgen", "encodec", "cosyvoice", "sparktts", "deltas"]

    for target in selected_targets:
        if target == "deltas":
            download_deltas(checkpoints_root / "finetunes")
            continue

        repo_id, relative_dir = TARGETS[target]
        # relative_dir is e.g. "checkpoints/moshi"; strip the "checkpoints/" prefix
        # so the destination is always checkpoints_root/<model_subdir>
        rel_subdir = Path(relative_dir).relative_to("checkpoints")
        download_repo(repo_id, checkpoints_root / rel_subdir)

    print("Done.")


if __name__ == "__main__":
    main()