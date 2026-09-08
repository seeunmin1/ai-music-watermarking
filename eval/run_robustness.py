#!/usr/bin/env python3
"""Mode B robustness harness: does the watermark survive the release chain?

This answers the "how do we survive producer audio perturbations?" question
directly. For each carrier track it embeds a known record ID, pushes the marked
audio through the perturbation battery, and tries to decode it back.

Two decoders are scored on identical audio so the report can attribute each
failure to the right cause:

  baseline    - the shipping decoder, which assumes the payload starts at
                sample 0.
  sync_search - a proposed decoder that locates the payload first.

Where `baseline` fails and `sync_search` succeeds, the watermark survived the
perturbation and the decoder simply lost alignment. That is a decoder fix, not
a watermark redesign. Where both fail, the signal itself did not survive.

Usage:
  python eval/run_robustness.py
  python eval/run_robustness.py --duration 30 --json eval/results/robustness.json
  python eval/run_robustness.py --carriers path/to/a.wav path/to/b.mp3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
CLI_DIR = EVAL_DIR.parents[0] / "cli"
for candidate in (str(CLI_DIR), str(EVAL_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from dataset import load_items  # noqa: E402
from detectors import DETECTORS  # noqa: E402
from perturbations import Perturbation, build_suite  # noqa: E402

import audiomark  # noqa: E402

# A fixed ID makes "did we decode the right payload?" checkable, and keeps runs
# comparable with each other.
TEST_RECORD_ID = 0xA5C3E1


def load_audio(path: Path, duration_s: float, rate_hint: int = 44100) -> tuple[np.ndarray, int]:
    """Decode any supported file to mono float64, trimmed to `duration_s`."""
    try:
        samples, rate, _ = audiomark.read_wav(path)
    except Exception:
        import miniaudio

        decoded = miniaudio.decode_file(str(path), output_format=miniaudio.SampleFormat.SIGNED16)
        rate = int(decoded.sample_rate or rate_hint)
        channels = int(getattr(decoded, "nchannels", 1) or 1)
        samples = np.asarray(decoded.samples, dtype=np.int16).astype(np.float64) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

    wanted = int(rate * duration_s)
    return samples[:wanted], rate


def embedding_snr(clean: np.ndarray, marked: np.ndarray) -> float:
    """Signal-to-watermark ratio in dB — how far under the music the mark sits."""
    length = min(len(clean), len(marked))
    residual = marked[:length] - clean[:length]
    signal_power = float(np.mean(clean[:length] ** 2))
    noise_power = float(np.mean(residual**2))
    if noise_power <= 0 or signal_power <= 0:
        return float("inf")
    return 10 * math.log10(signal_power / noise_power)


def run_carrier(
    path: Path,
    duration_s: float,
    perturbations: list[Perturbation],
    record_id: int,
) -> list[dict[str, Any]]:
    """Embed into one carrier, then score every perturbation and decoder."""
    clean, rate = load_audio(path, duration_s)
    if len(clean) < audiomark.FRAME:
        return [{
            "carrier": path.name,
            "perturbation": "-",
            "error": f"carrier shorter than one payload frame ({audiomark.FRAME} samples)",
        }]

    marked, repetitions = audiomark.embed_watermark(clean, record_id)
    if marked is None:
        return [{"carrier": path.name, "perturbation": "-", "error": "embedding failed"}]

    snr = embedding_snr(clean, marked)
    rows: list[dict[str, Any]] = []

    for perturbation in perturbations:
        started = time.time()
        try:
            attacked = perturbation.apply(marked, rate)
        except Exception as exc:
            rows.append({
                "carrier": path.name,
                "perturbation": perturbation.name,
                "category": perturbation.category,
                "severity": perturbation.severity,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        row: dict[str, Any] = {
            "carrier": path.name,
            "perturbation": perturbation.name,
            "category": perturbation.category,
            "severity": perturbation.severity,
            "detail": perturbation.detail,
            "embedding_snr_db": round(snr, 2),
            "repetitions_embedded": repetitions,
            "samples_in": len(marked),
            "samples_out": len(attacked),
            "apply_seconds": round(time.time() - started, 3),
        }
        for name, detector in DETECTORS.items():
            detection = detector(attacked, rate)
            row[f"{name}_found"] = bool(detection.found)
            row[f"{name}_id_correct"] = bool(detection.found and detection.record_id == record_id)
            row[f"{name}_z"] = round(float(detection.z), 3)
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-carrier results up to per-perturbation survival rates."""
    scored = [r for r in rows if "error" not in r]
    by_perturbation: dict[str, dict[str, Any]] = {}

    for row in scored:
        bucket = by_perturbation.setdefault(row["perturbation"], {
            "perturbation": row["perturbation"],
            "category": row["category"],
            "severity": row["severity"],
            "detail": row.get("detail", ""),
            "trials": 0,
        })
        bucket["trials"] += 1
        for name in DETECTORS:
            bucket[f"{name}_survived"] = bucket.get(f"{name}_survived", 0) + int(row[f"{name}_id_correct"])
            bucket.setdefault(f"{name}_z_values", []).append(row[f"{name}_z"])

    for bucket in by_perturbation.values():
        for name in DETECTORS:
            z_values = bucket.pop(f"{name}_z_values", []) or [0.0]
            bucket[f"{name}_rate"] = round(bucket[f"{name}_survived"] / bucket["trials"], 4)
            bucket[f"{name}_mean_z"] = round(float(np.mean(z_values)), 2)

    totals = {}
    for name in DETECTORS:
        survived = sum(int(r[f"{name}_id_correct"]) for r in scored)
        totals[name] = {
            "survived": survived,
            "trials": len(scored),
            "rate": round(survived / len(scored), 4) if scored else 0.0,
        }

    # Cases where the mark survived but the shipping decoder lost alignment.
    recoverable = [
        r["perturbation"] for r in scored
        if not r["baseline_id_correct"] and r.get("sync_search_id_correct")
    ]
    return {
        "totals": totals,
        "by_perturbation": sorted(by_perturbation.values(), key=lambda b: (b["baseline_rate"], b["perturbation"])),
        "recoverable_with_sync_search": sorted(set(recoverable)),
        "errors": [r for r in rows if "error" in r],
    }


def print_report(summary: dict[str, Any], rows: list[dict[str, Any]], carriers: list[Path], duration_s: float) -> None:
    bar = "=" * 92
    print(bar)
    print("  AUDIOMARK WATERMARK ROBUSTNESS (Mode B)")
    print(bar)
    print(f"  Carriers        : {len(carriers)} ({', '.join(c.name for c in carriers)})")
    print(f"  Segment length  : {duration_s:.0f} s per carrier")
    scored = [r for r in rows if "error" not in r]
    if scored:
        print(f"  Embedding SNR   : {scored[0]['embedding_snr_db']:.1f} dB "
              f"(watermark sits this far under the music)")
        print(f"  Payload reps    : {scored[0]['repetitions_embedded']}x")

    print("\n  OVERALL SURVIVAL (correct record ID recovered)")
    for name, totals in summary["totals"].items():
        print(f"    {name:<14} {totals['survived']:>3}/{totals['trials']:<3} = {totals['rate']:6.1%}")

    print(f"\n  {'PERTURBATION':<22}{'CATEGORY':<13}{'SEV':<10}{'BASELINE':>10}{'SYNC':>10}   {'BASE z':>8}{'SYNC z':>9}")
    print("  " + "-" * 88)
    for bucket in summary["by_perturbation"]:
        baseline = "PASS" if bucket["baseline_rate"] == 1 else ("FAIL" if bucket["baseline_rate"] == 0 else f"{bucket['baseline_rate']:.0%}")
        sync = "PASS" if bucket["sync_search_rate"] == 1 else ("FAIL" if bucket["sync_search_rate"] == 0 else f"{bucket['sync_search_rate']:.0%}")
        print(f"  {bucket['perturbation']:<22}{bucket['category']:<13}{bucket['severity']:<10}"
              f"{baseline:>10}{sync:>10}   {bucket['baseline_mean_z']:>8.2f}{bucket['sync_search_mean_z']:>9.2f}")

    if summary["recoverable_with_sync_search"]:
        print("\n  SURVIVED THE PERTURBATION BUT THE SHIPPING DECODER MISSED IT")
        print("  (the mark is still in the audio; the decoder lost alignment)")
        for name in summary["recoverable_with_sync_search"]:
            print(f"    - {name}")

    both_failed = [
        b["perturbation"] for b in summary["by_perturbation"]
        if b["baseline_rate"] == 0 and b["sync_search_rate"] == 0
    ]
    if both_failed:
        print("\n  WATERMARK DESTROYED (neither decoder recovered it)")
        for name in both_failed:
            print(f"    - {name}")

    if summary["errors"]:
        print("\n  ERRORS")
        for row in summary["errors"]:
            print(f"    {row['carrier']} / {row['perturbation']}: {row['error']}")
    print(bar)


def resolve_carriers(explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [Path(p).expanduser() for p in explicit]
    # Default to real audio from the validation set so the carrier is musical
    # rather than synthetic; the mark's gain rides on local signal level.
    items = load_items()
    paths = [i.path for i in items if i.path.exists()]
    repo_sample = EVAL_DIR.parents[0] / "test-audio" / "audiomark_source_5s.wav"
    if not paths and repo_sample.exists():
        paths = [repo_sample]
    return paths[:3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audiomark Mode B watermark robustness harness")
    parser.add_argument("--carriers", nargs="*", help="audio files to use as carriers")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds of each carrier to use")
    parser.add_argument("--record-id", type=lambda v: int(v, 0), default=TEST_RECORD_ID)
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument("--only", nargs="*", help="run only these perturbation names")
    args = parser.parse_args(argv)

    carriers = resolve_carriers(args.carriers)
    if not carriers:
        print("No carrier audio found. Pass --carriers explicitly.", file=sys.stderr)
        return 1

    suite = build_suite()
    if args.only:
        suite = [p for p in suite if p.name in set(args.only)]

    rows: list[dict[str, Any]] = []
    for carrier in carriers:
        rows.extend(run_carrier(carrier, args.duration, suite, args.record_id))

    summary = aggregate(rows)
    print_report(summary, rows, carriers, args.duration)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"summary": summary, "rows": rows,
                        "config": {"carriers": [str(c) for c in carriers],
                                   "duration_s": args.duration,
                                   "record_id": f"0x{args.record_id:06x}"}},
                       indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
