#!/usr/bin/env python3
"""Mode A accuracy harness: does detection reach the right statutory verdict?

Runs the shipping detector over the labeled validation set and scores it under
each verdict policy. The report separates two things that are easy to conflate:

  detection accuracy - of the files that carry a provenance payload, how many
                       did we read correctly? This is a property of our code.
  corpus coverage    - how many AI files carry a payload at all? This is a
                       property of the ecosystem. A file with no payload and no
                       decodable watermark cannot be detected by any
                       provenance-based tool, ours included.

Reporting only the first overstates the product; reporting only the second
hides real bugs. The summary prints both.

Usage:
  python eval/run_detection_eval.py
  python eval/run_detection_eval.py --json results/detection.json
  python eval/run_detection_eval.py --root downloads=/path/to/audio
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
CLI_DIR = EVAL_DIR.parents[0] / "cli"
for candidate in (str(CLI_DIR), str(EVAL_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from dataset import Item, cohort_summary, load_items, missing_items  # noqa: E402
from metrics import VERDICT_POLICIES, format_matrix, score_policy  # noqa: E402

import audiomark  # noqa: E402

# Verdicts that mean "we recovered an actual provenance payload from the file",
# as opposed to a bare vendor string appearing somewhere in the bytes.
PAYLOAD_VERDICTS = {"ai_generated", "c2pa_detected_untrusted", "probably_ai_generated"}


def analyze(item: Item) -> dict[str, Any]:
    """Run detection on one item and flatten what the scorer needs."""
    try:
        result = audiomark.op_detect(item.path)
    except Exception as exc:  # a decode failure is a result, not a crash
        return {
            "id": item.id,
            "path": str(item.path),
            "label": item.label,
            "cohort": item.cohort,
            "expected_provenance": item.expected_provenance,
            "verdict": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }

    provenance = result["provenance"]
    detection = result["detection"]
    metadata = result["metadata"]
    return {
        "id": item.id,
        "path": str(item.path),
        "label": item.label,
        "cohort": item.cohort,
        "expected_provider": item.provider,
        "expected_provenance": item.expected_provenance,
        "verdict": provenance["verdict"],
        "claim_level": provenance["claimLevel"],
        "declares_ai_generated": provenance.get("declaresAiGenerated", False),
        "display_title": provenance["displayTitle"],
        "detected_provider": provenance.get("attributedProvider"),
        "detected_system": provenance.get("attributedSystem"),
        "provider_verification": provenance["providerVerificationStatus"],
        "resolved_via": provenance.get("resolvedVia"),
        "confidence": provenance.get("confidence", 0),
        "manifest_found": bool(metadata.get("manifest")),
        "manifest_format": metadata.get("manifest_format"),
        "vendor_hints": metadata.get("vendor_hints", []),
        "watermark_found": bool(detection.found),
        "watermark_z": round(float(detection.z), 3),
        "audio_decoded": result["audio_decode"]["status"] == "decoded",
        "classifier_status": result["classifier"]["status"],
    }


def provider_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Among AI files where a provider was named, how often was it the right one?"""
    considered = [r for r in rows if r["label"] == "ai" and r.get("expected_provider")]
    named = [r for r in considered if r.get("detected_provider")]
    correct = [
        r for r in named
        if str(r["expected_provider"]).lower() in str(r["detected_provider"]).lower()
        or str(r["detected_provider"]).lower() in str(r["expected_provider"]).lower()
    ]
    return {
        "ai_files_with_known_provider": len(considered),
        "provider_named": len(named),
        "provider_correct": len(correct),
        "precision_when_named": round(len(correct) / len(named), 4) if named else None,
    }


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of the AI corpus carries a payload we could ever read."""
    ai_rows = [r for r in rows if r["label"] == "ai"]
    with_payload = [r for r in ai_rows if r["verdict"] in PAYLOAD_VERDICTS]
    hints_only = [r for r in ai_rows if r["verdict"] == "unknown_with_hints"]
    nothing = [r for r in ai_rows if r["verdict"] == "unmarked_or_unknown"]
    return {
        "ai_files": len(ai_rows),
        "payload_recovered": len(with_payload),
        "metadata_hint_only": len(hints_only),
        "no_signal": len(nothing),
        "payload_rate": round(len(with_payload) / len(ai_rows), 4) if ai_rows else None,
    }


def build_report(rows: list[dict[str, Any]], items: list[Item], missing: list[Item]) -> dict[str, Any]:
    scored = [r for r in rows if r["verdict"] != "error" and r["label"] in ("ai", "human")]
    assisted = [r for r in rows if r["label"] == "ai_assisted"]
    errors = [r for r in rows if r["verdict"] == "error"]

    policies = {name: score_policy(scored, name).to_dict() for name in VERDICT_POLICIES}
    return {
        "dataset": {
            "items_evaluated": len(rows),
            "scored_binary": len(scored),
            "ai_assisted_excluded": len(assisted),
            "errors": len(errors),
            "missing_from_disk": [i.id for i in missing],
            "cohorts": cohort_summary(items),
        },
        "policies": policies,
        "coverage": coverage(rows),
        "provider_attribution": provider_accuracy(rows),
        "verdict_distribution": _distribution(rows),
        "ai_assisted": assisted,
        "errors": errors,
        "rows": rows,
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        out.setdefault(row["label"], {}).setdefault(row["verdict"], 0)
        out[row["label"]][row["verdict"]] += 1
    return out


def print_report(report: dict[str, Any]) -> None:
    bar = "=" * 74
    dataset = report["dataset"]
    print(bar)
    print("  AUDIOMARK DETECTION ACCURACY (Mode A)")
    print(bar)
    print(f"  Files evaluated      : {dataset['items_evaluated']}")
    print(f"  Scored as binary     : {dataset['scored_binary']} (ai vs human)")
    print(f"  ai_assisted excluded : {dataset['ai_assisted_excluded']} (scored separately below)")
    if dataset["missing_from_disk"]:
        print(f"  Declared but missing : {', '.join(dataset['missing_from_disk'])}")
    print(f"  Cohorts              : {json.dumps(dataset['cohorts'])}")
    if dataset["errors"]:
        print(f"  Errors               : {dataset['errors']}")

    print("\n  ACCURACY BY VERDICT POLICY")
    print("  (which statutory claim levels are treated as a positive AI call)")
    for name in VERDICT_POLICIES:
        matrix = report["policies"][name]
        from metrics import ConfusionMatrix

        restored = ConfusionMatrix(
            matrix["true_positive"], matrix["false_positive"],
            matrix["true_negative"], matrix["false_negative"],
        )
        print(format_matrix(name, restored))

    cov = report["coverage"]
    print("\n  CORPUS COVERAGE (ecosystem property, not a code defect)")
    print(f"    AI files                  : {cov['ai_files']}")
    print(f"    Provenance payload read   : {cov['payload_recovered']}"
          f"  ({cov['payload_rate']:.0%})" if cov["payload_rate"] is not None else "")
    print(f"    Metadata hint only        : {cov['metadata_hint_only']}")
    print(f"    No signal at all          : {cov['no_signal']}")

    attribution = report["provider_attribution"]
    print("\n  PROVIDER ATTRIBUTION")
    print(f"    AI files w/ known provider: {attribution['ai_files_with_known_provider']}")
    print(f"    Provider named            : {attribution['provider_named']}")
    print(f"    Named correctly           : {attribution['provider_correct']}")

    print("\n  VERDICT DISTRIBUTION")
    for label, counts in report["verdict_distribution"].items():
        print(f"    {label}")
        for verdict, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"      {verdict:<32} {count}")

    if report["ai_assisted"]:
        print("\n  AI-ASSISTED (reported partial AI use; scored separately)")
        for row in report["ai_assisted"]:
            print(f"    {row['id']:<40} {row['verdict']}")

    print("\n  PER-FILE RESULTS")
    for row in report["rows"]:
        if row["verdict"] == "error":
            print(f"    {row['id']:<40} ERROR  {row['error']}")
            continue
        provider = row.get("detected_provider") or "-"
        print(f"    {row['id']:<40} {row['label']:<12} {row['verdict']:<28} "
              f"{provider:<14} z={row['watermark_z']:.2f}")
    print(bar)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audiomark Mode A detection accuracy harness")
    parser.add_argument("--labels", type=Path, default=EVAL_DIR / "labels.json")
    parser.add_argument("--root", action="append", default=[],
                        metavar="NAME=PATH", help="override a labels.json root")
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    overrides = {}
    for pair in args.root:
        name, _, value = pair.partition("=")
        overrides[name] = value

    items = load_items(args.labels, overrides)
    missing = missing_items(args.labels, overrides)
    if not items:
        print("No labeled audio found on disk. Check labels.json roots.", file=sys.stderr)
        return 1

    rows = [analyze(item) for item in items]
    report = build_report(rows, items, missing)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
