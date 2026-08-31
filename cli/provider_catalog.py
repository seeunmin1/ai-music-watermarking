"""Dynamic provider profile loader for generative audio provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROVIDERS_DIR = Path(__file__).resolve().parents[1] / "providers"


def load_provider_profiles(providers_dir: Path = PROVIDERS_DIR) -> list[dict[str, Any]]:
    profiles = []
    if not providers_dir.exists():
        return profiles
    for path in sorted(providers_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_profile_path"] = str(path)
        profiles.append(data)
    return profiles


def provider_aliases(profiles: list[dict[str, Any]] | None = None) -> dict[str, str]:
    out = {}
    for profile in profiles or load_provider_profiles():
        for alias in profile.get("aliases", []):
            out[alias.lower()] = profile["display_name"]
    return out


def vendor_labels_from_catalog() -> dict[str, str]:
    labels = {}
    for profile in load_provider_profiles():
        for alias in profile.get("aliases", []):
            labels[alias.lower()] = profile["display_name"]
    return labels


def profile_by_id(provider_id: str) -> dict[str, Any] | None:
    for profile in load_provider_profiles():
        if profile.get("provider_id") == provider_id:
            return profile
    return None
