#!/usr/bin/env python3
"""Train the local probable-attribution model from labeled WAV folders.

Expected layout:
  dataset_dir/
    Google/*.wav
    OpenAI/*.wav
    ElevenLabs/*.wav
    Suno/*.wav
    Human/*.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

from local_attribution import DEFAULT_MODEL_PATH, train_centroid_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local probable audio attribution model")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    model = train_centroid_model(args.dataset_dir, args.output)
    for provider, rec in model["providers"].items():
        print(f"{provider}: {rec['samples']} samples")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
