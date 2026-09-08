#!/usr/bin/env python3
"""Export the validation set as a shareable, reproducible package.

Produces a flat table joining three things that otherwise live apart: the
ground-truth label, where the audio came from, and what detection returned for
it. That is what someone needs to audit a result, disagree with a label, or
rerun the numbers.

Audio is *not* bundled by default. The AI tracks come from SONICS and the human
tracks from FMA, each under its own terms, so the package ships a manifest and
a seeded rebuild command instead of redistributing the corpus. `--with-audio`
bundles it anyway for internal sharing.

Usage:
  python eval/export_dataset.py
  python eval/export_dataset.py --with-audio --dest ~/audiomark-validation-set
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
CLI_DIR = EVAL_DIR.parents[0] / "cli"
for candidate in (str(CLI_DIR), str(EVAL_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from dataset import load_items  # noqa: E402

DEFAULT_RESULTS = EVAL_DIR / "results" / "detection_full.json"
CORPUS_MANIFEST = Path("~/ai-music-corpus/corpus_manifest.json").expanduser()

COLUMNS = [
    "id", "cohort", "label", "expected_provider", "expected_provenance",
    "verdict", "detected_provider", "provider_verification", "correct",
    "manifest_found", "manifest_format", "watermark_z", "vendor_hints",
    "source_dataset", "source_citation", "source_archive", "filename",
    "sha256", "bytes", "label_confidence", "notes",
]

# A verdict counts as an AI call under the disclosure policy; `ai_assisted` is
# scored separately everywhere, so it has no correct/incorrect binary answer.
AI_VERDICTS = {"ai_generated", "c2pa_detected_untrusted", "probably_ai_generated"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_index() -> dict[str, dict[str, Any]]:
    """Map filename -> where the corpus fetcher got it."""
    if not CORPUS_MANIFEST.exists():
        return {}
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    return {record["file"]: record for record in manifest.get("files", [])}


def build_rows(results_path: Path, hash_files: bool) -> list[dict[str, Any]]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in results["rows"]}
    sources = load_source_index()
    items = {item.id: item for item in load_items()}

    rows: list[dict[str, Any]] = []
    for item_id, item in items.items():
        result = by_id.get(item_id, {})
        source = sources.get(item.path.name, {})
        verdict = result.get("verdict", "not_evaluated")

        if item.label == "ai_assisted":
            correct = ""  # scored separately, no binary ground truth
        else:
            predicted_ai = verdict in AI_VERDICTS
            correct = str(predicted_ai == (item.label == "ai")).lower()

        rows.append({
            "id": item_id,
            "cohort": item.cohort,
            "label": item.label,
            "expected_provider": item.provider or "",
            "expected_provenance": item.expected_provenance,
            "verdict": verdict,
            "detected_provider": result.get("detected_provider") or "",
            "provider_verification": result.get("provider_verification") or "",
            "correct": correct,
            "manifest_found": str(bool(result.get("manifest_found"))).lower(),
            "manifest_format": result.get("manifest_format") or "",
            "watermark_z": result.get("watermark_z", ""),
            "vendor_hints": ";".join(result.get("vendor_hints") or []),
            "source_dataset": source.get("dataset", item.source or "supplied by the team"),
            "source_citation": source.get("citation", ""),
            "source_archive": source.get("source_archive", ""),
            "filename": item.path.name,
            "sha256": sha256(item.path) if hash_files and item.path.exists() else "",
            "bytes": item.path.stat().st_size if item.path.exists() else "",
            "label_confidence": item.label_confidence,
            "notes": (item.notes or source.get("note", "")).replace("\n", " "),
        })

    rows.sort(key=lambda r: (r["cohort"], r["id"]))
    return rows


def write_readme(dest: Path, rows: list[dict[str, Any]], with_audio: bool) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["cohort"], row["label"])
        counts[key] = counts.get(key, 0) + 1
    table = "\n".join(
        f"| {cohort} | {label} | {count} |" for (cohort, label), count in sorted(counts.items())
    )
    correct = sum(1 for r in rows if r["correct"] == "true")
    scored = sum(1 for r in rows if r["correct"] in ("true", "false"))
    detected = sum(1 for r in rows if r["label"] == "ai" and r["correct"] == "true")
    ai_total = sum(1 for r in rows if r["label"] == "ai")

    audio_section = (
        "Audio is included under `audio/<cohort>/`.\n"
        if with_audio
        else (
            "Audio is **not** included. Rebuild the public portion exactly:\n\n"
            "```bash\n"
            "python eval/fetch_corpus.py --suno 40 --udio 40 --human 40 --seed 7\n"
            "```\n\n"
            "The Gemini, Chopin and game tracks came from the team and are not public.\n"
        )
    )

    dest.joinpath("README.md").write_text(f"""# Audiomark validation set

{len(rows)} labeled tracks used to measure detection accuracy.

| Cohort | Label | n |
| --- | --- | --- |
{table}

Detection reached the right answer on **{correct}/{scored}** scored tracks
({detected}/{ai_total} AI tracks detected). `ai_assisted` is excluded from that
binary because a partly-AI track is a different statutory question.

## Files

- `validation_set.csv` — one row per track: label, source, and detection result.
- `validation_set.json` — the same rows, plus run metadata.
- `labels.json` — the ground-truth definition the harness reads.

## Columns worth knowing

| Column | Meaning |
| --- | --- |
| `label` | ground truth: `ai`, `human`, or `ai_assisted` |
| `expected_provenance` | what the file should carry: `c2pa_signed`, `none`, `unknown` |
| `verdict` | what detection returned |
| `correct` | did the verdict match the label, under the disclosure policy |
| `manifest_format` | `jumbf_cbor` = real C2PA; `audiomark_json` = our own |
| `watermark_z` | Audiomark watermark correlation; detection threshold is 2.5 |
| `label_confidence` | `confirmed` = verified by inspection, `reported` = told to us |

## The caveat that matters

The Suno and Udio tracks come from **SONICS**, where the authors re-encoded
everything with LAME 3.100 and stripped the ID3 tags. Their 0% detection rate
shows **what survives collection and redistribution** — not that Suno and Udio
failed to disclose at generation time. Answering that needs first-party
downloads.

The Gemini tracks, by contrast, were downloaded straight from the generator and
still carry their signed manifests. That contrast is the finding.

## Sources

- **SONICS** — *Synthetic Or Not: Identifying Counterfeit Songs*, ICLR 2025.
  `huggingface.co/datasets/awsaf49/sonics`
- **FMA** — Defferrard et al., *FMA: A Dataset For Music Analysis*, ISMIR 2017.
  `github.com/mdeff/fma`
- Gemini / Chopin / game tracks — supplied by the team.

{audio_section}
Each cohort keeps its source dataset's terms. Check those before redistributing.
""", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the labeled validation set")
    parser.add_argument("--dest", type=Path, default=Path("~/audiomark-validation-set").expanduser())
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--with-audio", action="store_true", help="bundle the audio files too")
    parser.add_argument("--no-hash", action="store_true", help="skip sha256 (faster)")
    parser.add_argument("--zip", action="store_true", help="also produce a .zip of the package")
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"No results at {args.results}. Run run_detection_eval.py first.", file=sys.stderr)
        return 1

    dest = args.dest.expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.results, hash_files=not args.no_hash)

    with dest.joinpath("validation_set.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    dest.joinpath("validation_set.json").write_text(json.dumps({
        "count": len(rows),
        "results_source": str(args.results),
        "rebuild": "python eval/fetch_corpus.py --suno 40 --udio 40 --human 40 --seed 7",
        "rows": rows,
    }, indent=2), encoding="utf-8")

    shutil.copy(EVAL_DIR / "labels.json", dest / "labels.json")

    if args.with_audio:
        items = {item.id: item for item in load_items()}
        for row in rows:
            item = items.get(row["id"])
            if not item or not item.path.exists():
                continue
            # Group by generator rather than cohort: browsing "suno" and "gemini"
            # is far more useful than one bucket holding every AI track.
            group = (row["expected_provider"] or row["label"]).lower().replace(" ", "_")
            folder = dest / "audio" / group
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(item.path, folder / item.path.name)

    write_readme(dest, rows, args.with_audio)

    if args.zip:
        archive = shutil.make_archive(str(dest), "zip", root_dir=dest)
        print(f"  zip: {archive} ({Path(archive).stat().st_size / 1e6:.1f} MB)")

    print(f"Exported {len(rows)} tracks to {dest}")
    for name in sorted(p.name for p in dest.iterdir()):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
