"""Local C2PA/Content Credentials matching against provider profiles.

This verifier works with the JSON manifests already extracted by the MVP
scanner. It is not a complete C2PA cryptographic validator yet; it separates
issuer/profile matching from signature-chain validation so the latter can be
added with a dedicated C2PA library without changing result shape.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _manifest_text(manifest: dict[str, Any], fields: dict[str, Any] | None) -> str:
    return json.dumps({"manifest": manifest, "fields": fields or {}}, default=str).lower()


def _includes_any(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text) for term in terms)


def verify_c2pa_manifest(meta: dict[str, Any], provider_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = meta.get("manifest")
    if not manifest:
        return []
    fields = meta.get("fields") or {}
    text = _manifest_text(manifest, fields)
    results = []

    for profile in provider_profiles:
        issuers = profile.get("c2pa_issuers", [])
        aliases = profile.get("aliases", [])
        if not issuers and "c2pa" not in profile.get("signals", []):
            continue
        if not (_includes_any(text, issuers) or _includes_any(str(fields.get("provider", "")), aliases)):
            continue

        policy = profile.get("confidence_policy", {})
        results.append({
            "id": f"c2pa:{profile['provider_id']}",
            "provider": profile["display_name"],
            "label": f"{profile['display_name']} C2PA Content Credentials",
            "configured": True,
            "verified": True,
            "status": "verified",
            "confidence": policy.get("trusted_c2pa", 94),
            "system": fields.get("system") or (profile.get("products") or [profile["display_name"]])[0],
            "contentId": fields.get("uid") or fields.get("unique_id"),
            "createdAt": fields.get("created"),
            "detail": "Provider issuer matched the extracted C2PA-style manifest; cryptographic chain validation is pending.",
            "evidence": {
                "profile": profile["provider_id"],
                "matched": "c2pa_issuer_or_provider_field",
                "signatureValidation": "pending",
            },
        })
    return results
