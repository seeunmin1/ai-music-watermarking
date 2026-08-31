"""Official provider verification adapter boundary.

Provider checks are loaded from providers/*.json. Adapters never manufacture a
verified result: a profile is verified only through a successful official API
response or local C2PA issuer match.
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from c2pa_verifier import verify_c2pa_manifest
from provider_catalog import load_provider_profiles


def _status_for_unconfigured(profile: dict[str, Any]) -> dict[str, Any]:
    official = profile.get("official_verification", {})
    env_key = official.get("env_key")
    env_url = official.get("env_url")
    needed = [x for x in (env_key, env_url) if x]
    suffix = f"; set {', '.join(needed)} to enable" if needed else ""
    return {
        "id": f"official:{profile['provider_id']}",
        "provider": profile["display_name"],
        "label": f"{profile['display_name']} official check",
        "configured": False,
        "verified": False,
        "status": "not_configured" if official.get("status") in {"available", "manual", "partner_only"} else "unsupported",
        "detail": f"{profile['display_name']} official verification is {official.get('status', 'unknown')}{suffix}.",
        "evidence": {"profile": profile["provider_id"], "adapter": official.get("adapter", "manual")},
    }


def _parse_openai_response(profile: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    detected = None
    for item in payload.get("results", []):
        if item.get("outcome") == "detected":
            detected = item
            break
    if not detected:
        return {
            "id": "official:openai",
            "provider": profile["display_name"],
            "label": "OpenAI Content Provenance API",
            "configured": True,
            "verified": False,
            "status": "not_detected",
            "detail": "OpenAI API did not detect supported provenance signals.",
            "evidence": payload,
        }
    return {
        "id": "official:openai",
        "provider": profile["display_name"],
        "label": "OpenAI Content Provenance API",
        "configured": True,
        "verified": True,
        "status": "verified",
        "confidence": profile.get("confidence_policy", {}).get("official_verified", 98),
        "system": detected.get("model") or profile.get("system") or "OpenAI generated audio",
        "contentId": detected.get("content_id"),
        "createdAt": detected.get("generated_at"),
        "detail": "OpenAI official provenance API detected a supported signal.",
        "evidence": payload,
    }


def _call_openai(profile: dict[str, Any], path: str) -> dict[str, Any]:
    official = profile.get("official_verification", {})
    api_key = os.getenv(official.get("env_key", "OPENAI_API_KEY"))
    if not api_key:
        return _status_for_unconfigured(profile)

    url = os.getenv(official.get("env_url", "")) or official.get("default_url")
    boundary = f"audiomark-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    file_bytes = Path(path).read_bytes()
    filename = Path(path).name
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {mime}\r\n\r\n".encode(),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            payload = json.loads(res.read().decode("utf-8"))
        return _parse_openai_response(profile, payload)
    except urllib.error.HTTPError as exc:
        status = "rate_limited" if exc.code == 429 else "error"
        return {
            "id": "official:openai",
            "provider": profile["display_name"],
            "label": "OpenAI Content Provenance API",
            "configured": True,
            "verified": False,
            "status": status,
            "detail": f"OpenAI official check failed with HTTP {exc.code}.",
            "evidence": {"httpStatus": exc.code},
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "id": "official:openai",
            "provider": profile["display_name"],
            "label": "OpenAI Content Provenance API",
            "configured": True,
            "verified": False,
            "status": "error",
            "detail": f"OpenAI official check failed: {exc}",
        }


def _run_remote_placeholder(profile: dict[str, Any]) -> dict[str, Any]:
    official = profile.get("official_verification", {})
    url = os.getenv(official.get("env_url", ""))
    key = os.getenv(official.get("env_key", ""))
    if not (url or key):
        return _status_for_unconfigured(profile)
    return {
        "id": f"official:{profile['provider_id']}",
        "provider": profile["display_name"],
        "label": f"{profile['display_name']} official check",
        "configured": True,
        "verified": False,
        "status": "configured_manual_adapter",
        "detail": f"{profile['display_name']} credentials are configured; provider-specific network adapter is pending.",
        "evidence": {"profile": profile["provider_id"], "adapter": official.get("adapter", "manual")},
    }


def run_official_checks(path: str, meta: dict[str, Any], profiles: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    profiles = profiles or load_provider_profiles()
    results = verify_c2pa_manifest(meta, profiles)
    matched_ids = {r.get("evidence", {}).get("profile") for r in results}

    for profile in profiles:
        if profile["provider_id"] in matched_ids:
            continue
        adapter = profile.get("official_verification", {}).get("adapter", "manual")
        if adapter == "openai":
            results.append(_call_openai(profile, path))
        elif adapter == "c2pa_local":
            results.append(_status_for_unconfigured(profile))
        else:
            results.append(_run_remote_placeholder(profile))
    return results
