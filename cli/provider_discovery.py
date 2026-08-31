#!/usr/bin/env python3
"""Provider catalog discovery/report job.

This is a curation scaffold: it reports provider coverage, source URLs, and
integration status from providers/*.json. Automated crawling can update these
profiles later, but the detector should depend only on reviewed catalog files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from provider_catalog import load_provider_profiles


def build_report() -> dict:
    providers = []
    for profile in load_provider_profiles():
        official = profile.get("official_verification", {})
        providers.append({
            "provider_id": profile["provider_id"],
            "display_name": profile["display_name"],
            "signals": profile.get("signals", []),
            "official_status": official.get("status", "unknown"),
            "portal": official.get("portal"),
            "adapter": official.get("adapter", "manual"),
            "profile_path": profile.get("_profile_path"),
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_count": len(providers),
        "providers": providers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report provider catalog coverage")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
