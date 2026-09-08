#!/usr/bin/env python3
"""Build a labeled validation corpus from public datasets.

Fills the provider gap the first validation run exposed: every AI track we had
was Gemini, which happens to have the cleanest C2PA implementation in the
industry, so it told us nothing about generators that ship no manifest.

Sources:

  SONICS (awsaf49/sonics, ICLR 2025) - 25.4k Suno and 23.6k Udio songs.
  FMA small (Free Music Archive)      - 8k human tracks across 8 balanced genres.

Both ship as multi-gigabyte ZIPs, and a validation sample needs tens of tracks,
so members are pulled individually over HTTP range requests instead of
downloading ~45 GB of archives. Sampling is seeded, so a given seed reproduces
the same corpus.

**Read this before interpreting results on these files.** SONICS tracks were
collected and re-encoded by the dataset authors. If a Suno track arrives with no
C2PA manifest, that shows the manifest does not survive collection and
redistribution - not that Suno never wrote one. It is the right test for "AI
music encountered in the wild", and the wrong test for "does this provider
disclose at source".

Usage:
  python eval/fetch_corpus.py --suno 40 --udio 40 --human 40
  python eval/fetch_corpus.py --dest ~/ai-music-corpus --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from remote_zip import open_remote_zip  # noqa: E402

SONICS_PART = "https://huggingface.co/datasets/awsaf49/sonics/resolve/main/fake_songs/part_%02d.zip"
FMA_SMALL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"

# Part 4 carries a healthy mix of both generators, so one archive open serves
# both cohorts.
SONICS_DEFAULT_PART = 4

SOURCES = {
    "suno": {
        "dataset": "SONICS (awsaf49/sonics)",
        "citation": "SONICS: Synthetic Or Not - Identifying Counterfeit Songs, ICLR 2025",
        "label": "ai",
        "provider": "Suno",
        "note": "Suno output collected and re-encoded by the dataset authors.",
    },
    "udio": {
        "dataset": "SONICS (awsaf49/sonics)",
        "citation": "SONICS: Synthetic Or Not - Identifying Counterfeit Songs, ICLR 2025",
        "label": "ai",
        "provider": "Udio",
        "note": "Udio output collected and re-encoded by the dataset authors.",
    },
    "human": {
        "dataset": "FMA small (Free Music Archive)",
        "citation": "Defferrard et al., FMA: A Dataset For Music Analysis, ISMIR 2017",
        "label": "human",
        "provider": None,
        "note": "Creative Commons human recordings, 8 balanced genres, 30 s clips.",
    },
}


def fetch_sonics(dest: Path, want_suno: int, want_udio: int, seed: int, part: int) -> list[dict]:
    if want_suno <= 0 and want_udio <= 0:
        return []
    url = SONICS_PART % part
    print(f"  opening SONICS part_{part:02d} ...", flush=True)
    started = time.time()
    archive = open_remote_zip(url)
    names = [n for n in archive.namelist() if n.endswith(".mp3")]
    print(f"  indexed {len(names)} tracks in {time.time() - started:.1f}s", flush=True)

    rng = random.Random(seed)
    records = []
    for provider, want in (("suno", want_suno), ("udio", want_udio)):
        pool = [n for n in names if provider in n.lower()]
        rng.shuffle(pool)
        chosen = pool[:want]
        folder = dest / provider
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  {provider}: fetching {len(chosen)} of {len(pool)} available", flush=True)
        for index, name in enumerate(chosen, 1):
            target = folder / Path(name).name
            if not target.exists():
                with archive.open(name) as source:
                    target.write_bytes(source.read())
            records.append({
                "cohort": provider,
                "file": target.name,
                "source_member": name,
                "source_archive": url,
                **{k: v for k, v in SOURCES[provider].items()},
            })
            if index % 10 == 0 or index == len(chosen):
                print(f"    {index}/{len(chosen)}", flush=True)
    return records


def fetch_fma(dest: Path, want: int, seed: int) -> list[dict]:
    if want <= 0:
        return []
    print("  opening FMA small ...", flush=True)
    started = time.time()
    archive = open_remote_zip(FMA_SMALL)
    names = [n for n in archive.namelist() if n.endswith(".mp3")]
    print(f"  indexed {len(names)} tracks in {time.time() - started:.1f}s", flush=True)

    rng = random.Random(seed + 1)
    rng.shuffle(names)
    folder = dest / "human"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    print(f"  human: fetching {want} of {len(names)} available", flush=True)
    for index, name in enumerate(names[:want], 1):
        target = folder / Path(name).name
        if not target.exists():
            try:
                with archive.open(name) as source:
                    target.write_bytes(source.read())
            except Exception as exc:  # a few FMA members are known to be truncated
                print(f"    skipped {name}: {exc}", flush=True)
                continue
        records.append({
            "cohort": "human",
            "file": target.name,
            "source_member": name,
            "source_archive": FMA_SMALL,
            **{k: v for k, v in SOURCES["human"].items()},
        })
        if index % 10 == 0 or index == want:
            print(f"    {index}/{want}", flush=True)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a labeled AI/human music validation corpus")
    parser.add_argument("--dest", type=Path, default=Path("~/ai-music-corpus").expanduser())
    parser.add_argument("--suno", type=int, default=40)
    parser.add_argument("--udio", type=int, default=40)
    parser.add_argument("--human", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sonics-part", type=int, default=SONICS_DEFAULT_PART)
    args = parser.parse_args(argv)

    dest = args.dest.expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Building corpus at {dest}")

    records: list[dict] = []
    records.extend(fetch_sonics(dest, args.suno, args.udio, args.seed, args.sonics_part))
    records.extend(fetch_fma(dest, args.human, args.seed))

    manifest = dest / "corpus_manifest.json"
    manifest.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "sources": SOURCES,
        "caveat": (
            "SONICS tracks were collected and re-encoded by the dataset authors. "
            "Absent provenance shows what survives redistribution, not what the "
            "provider wrote at generation time."
        ),
        "files": records,
    }, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for record in records:
        counts[record["cohort"]] = counts.get(record["cohort"], 0) + 1
    print(f"\nDone. {len(records)} files: {counts}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
