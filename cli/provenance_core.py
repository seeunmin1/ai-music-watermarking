"""Normalized provenance orchestration for Audiomark CLI/backend detection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from provider_catalog import load_provider_profiles, vendor_labels_from_catalog

SHARED_REGISTRY = Path(__file__).resolve().parents[1] / "shared" / "provider_provenance.json"
PROVIDER_PROVENANCE = json.loads(SHARED_REGISTRY.read_text(encoding="utf-8"))
PROFILES = PROVIDER_PROVENANCE["profiles"]
VENDOR_LABELS = {**PROVIDER_PROVENANCE["vendor_labels"], **vendor_labels_from_catalog()}
LEGACY_PROVIDER_SIGNAL_IDS = {
    "google": "google_synthid",
    "openai": "openai_synthid",
    "elevenlabs": "elevenlabs_watermark",
    "suno": "suno_c2pa",
}


def _includes_any(value: str, terms: list[str]) -> bool:
    text = (value or "").lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text) for term in terms)


def _manifest_text(meta: dict[str, Any]) -> str:
    if not meta.get("manifest") and not meta.get("fields"):
        return ""
    return json.dumps({"manifest": meta.get("manifest"), "fields": meta.get("fields")}, default=str).lower()


def _claim_level(status: str) -> str:
    if status == "verified":
        return "verified"
    if status == "probable":
        return "probable"
    return "unknown"


def _manifest_signal(meta: dict[str, Any]) -> dict[str, Any] | None:
    if not meta.get("manifest"):
        return None
    fields = meta.get("fields") or {}
    return {
        "id": "c2pa_manifest",
        "label": PROFILES["c2pa_manifest"]["label"],
        "category": "Verified provenance",
        "status": "verified",
        "claimLevel": "verified",
        "confidence": 92,
        "provider": fields.get("provider"),
        "system": fields.get("system"),
        "contentId": fields.get("uid") or fields.get("unique_id"),
        "createdAt": fields.get("created"),
        "detail": "Embedded C2PA-style disclosure manifest parsed",
    }


def _provider_payload_signals(meta: dict[str, Any]) -> list[dict[str, Any]]:
    text = _manifest_text(meta)
    fields = meta.get("fields") or {}
    hints = set(meta.get("vendor_hints") or [])
    out = []

    for profile in load_provider_profiles():
        provider = profile["display_name"]
        signal_id = LEGACY_PROVIDER_SIGNAL_IDS.get(profile["provider_id"], f"provider_payload:{profile['provider_id']}")
        provider_text = str(fields.get("provider") or "").lower()
        provider_terms = [provider, *profile.get("aliases", [])]
        provider_match = provider_text and _includes_any(provider_text, provider_terms)
        issuer_match = _includes_any(text, profile.get("c2pa_issuers", []))
        hint_key = provider.lower().replace(" ", "")
        hint_only = any(v in hints for v in profile.get("aliases", []))
        terms = profile.get("aliases", []) + profile.get("products", []) + profile.get("signals", [])
        if text and _includes_any(text, terms) and (provider_match or issuer_match):
            out.append({
                "id": signal_id,
                "label": PROFILES.get(signal_id, {}).get("label", f"{provider} provider payload"),
                "category": "Verified provenance",
                "status": "verified",
                "claimLevel": "verified",
                "confidence": profile.get("confidence_policy", {}).get("trusted_c2pa", 94),
                "provider": provider,
                "system": fields.get("system") or (profile.get("products") or [provider])[0],
                "contentId": fields.get("uid") or fields.get("unique_id"),
                "createdAt": fields.get("created"),
                "detail": f"{provider} provenance payload recovered from metadata",
                "evidence": {"profile": profile["provider_id"], "source": "provider_catalog"},
            })
        elif hint_only:
            out.append({
                "id": f"metadata_vendor_hint:{hint_key}",
                "label": f"{provider}-related metadata",
                "category": "Metadata hint",
                "status": "hint",
                "claimLevel": "unknown",
                "confidence": 20,
                "provider": provider,
                "system": "Unknown",
                "contentId": None,
                "createdAt": None,
                "detail": f"{provider} text was present, but no trusted payload or official result was recovered",
                "evidence": {"profile": profile["provider_id"], "source": "metadata_hint"},
            })
    return out


def _official_result_signals(official_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for result in official_results or []:
        provider = result.get("provider")
        verified = bool(result.get("verified"))
        signal_id = result.get("id") or f"official:{str(provider or 'unknown').lower()}"
        out.append({
            "id": signal_id,
            "label": result.get("label") or f"{provider} official check",
            "category": "Verified provenance" if verified else "Unsupported/unknown",
            "status": "verified" if verified else result.get("status", "not_configured"),
            "claimLevel": "verified" if verified else "unknown",
            "confidence": result.get("confidence", 98 if verified else 0),
            "provider": provider if verified else None,
            "system": result.get("system"),
            "contentId": result.get("contentId"),
            "createdAt": result.get("createdAt"),
            "detail": result.get("detail") or "Official provider verification adapter result",
            "evidence": result.get("evidence"),
        })
    return out


def _audiomark_signal(det: Any, rec: dict[str, Any] | None) -> dict[str, Any] | None:
    if not getattr(det, "found", False):
        return None
    registered = rec is not None
    return {
        "id": "audiomark_ss_v1",
        "label": PROFILES["audiomark_ss_v1"]["label"],
        "category": "Verified provenance" if registered else "Detected watermark",
        "status": "verified" if registered else "detected",
        "claimLevel": "verified" if registered else "unknown",
        "confidence": getattr(det, "confidence", 0.0),
        "provider": rec["provider"] if registered else "Unknown",
        "system": f"{rec['system']} v{rec['version']}" if registered else "Unknown",
        "contentId": rec["uid"] if registered else f"record 0x{det.record_id:06x}",
        "createdAt": rec["timestamp"] if registered else None,
        "detail": "Latent watermark matched a local registry record" if registered else "Latent Audiomark watermark decoded without a registry match",
        "recordStatus": rec.get("status") if registered else None,
    }


def _classifier_signal(classifier_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not classifier_result or classifier_result.get("status") != "probable":
        return None
    top = (classifier_result.get("candidates") or [{}])[0]
    provider = top.get("provider")
    confidence = round(float(top.get("confidence", 0)) * 100)
    if not provider or confidence <= 0:
        return None
    return {
        "id": "local_attribution_classifier",
        "label": PROFILES["local_attribution_classifier"]["label"],
        "category": "Probable attribution",
        "status": "probable",
        "claimLevel": "probable",
        "confidence": confidence,
        "provider": provider,
        "system": top.get("system") or "Unknown",
        "contentId": None,
        "createdAt": None,
        "detail": classifier_result.get("detail") or "Local ML attribution model result; not verified provenance",
    }


def _vendor_hint_signals(meta: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    provider_hint_prefixes = set(vendor_labels_from_catalog()) | {"synthid", "audioseal"}
    for v in meta.get("vendor_hints") or []:
        if v in provider_hint_prefixes:
            continue
        label = VENDOR_LABELS.get(v, v)
        out.append({
            "id": f"metadata_vendor_hint:{v}",
            "label": label,
            "category": "Metadata hint",
            "status": "hint",
            "claimLevel": "unknown",
            "confidence": 15,
            "provider": label,
            "system": "Unknown",
            "contentId": None,
            "createdAt": None,
            "detail": "Vendor string appeared in metadata bytes; not a verified provenance payload",
        })
    return out


def _wmar_signal() -> dict[str, Any]:
    profile = PROFILES["wmar_clustered_token"]
    return {
        "id": "wmar_clustered_token",
        "label": "WMAR clustered-token detector",
        "category": "Unsupported/unknown",
        "status": "unsupported",
        "claimLevel": "unknown",
        "confidence": 0,
        "provider": None,
        "system": None,
        "contentId": None,
        "createdAt": None,
        "detail": f"Experimental backend hook only; upstream {profile['upstream']}",
    }


def _pick_primary(signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    for signal in signals:
        if signal["status"] == "verified" and (signal["id"].startswith("official:") or signal["id"].startswith("c2pa:")):
            return signal
    for signal in signals:
        if signal["status"] == "verified" and signal["id"].startswith("provider_payload:"):
            return signal
    for wanted in LEGACY_PROVIDER_SIGNAL_IDS.values():
        for signal in signals:
            if signal["id"] == wanted and signal["status"] == "verified":
                return signal
    for wanted in ("audiomark_ss_v1", "c2pa_manifest"):
        for signal in signals:
            if signal["id"] == wanted and signal["status"] == "verified":
                return signal
    for signal in signals:
        if signal["status"] == "probable":
            return signal
    return None


def _official_check_status(signals: list[dict[str, Any]], official_results: list[dict[str, Any]]) -> dict[str, Any]:
    official_only = [r for r in official_results or [] if str(r.get("id", "")).startswith("official:")]
    configured = [r for r in official_only if r.get("configured")]
    verified = [s for s in signals if s["status"] == "verified" and s["id"].startswith("official:")]
    if verified:
        status = "verified"
    elif configured:
        status = "checked_no_match"
    else:
        status = "not_configured"
    return {
        "status": status,
        "configuredProviders": [r.get("provider") for r in configured if r.get("provider")],
        "checkedProviders": [r.get("provider") for r in official_only if r.get("provider")],
    }


def analyze_provenance(
    det: Any,
    meta: dict[str, Any],
    registry: list[dict[str, Any]],
    official_results: list[dict[str, Any]] | None = None,
    classifier_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = next((r for r in registry if int(r["id_hex"], 16) == det.record_id), None) if det.found else None
    signals = [
        *_official_result_signals(official_results or []),
        _manifest_signal(meta),
        *_provider_payload_signals(meta),
        _audiomark_signal(det, rec),
        _classifier_signal(classifier_result),
        *_vendor_hint_signals(meta),
        _wmar_signal(),
    ]
    signals = [s for s in signals if s is not None]
    primary = _pick_primary(signals)
    has_hints = any(s["status"] == "hint" for s in signals)
    claim_level = _claim_level(primary["status"]) if primary else "unknown"

    if claim_level == "verified":
        verdict = "ai_generated"
    elif claim_level == "probable":
        verdict = "probably_ai_generated"
    elif has_hints:
        verdict = "unknown_with_hints"
    else:
        verdict = "unmarked_or_unknown"

    candidates = []
    for signal in signals:
        if signal.get("provider") and signal["status"] in {"verified", "probable", "hint"}:
            candidates.append({
                "provider": signal["provider"],
                "system": signal.get("system"),
                "confidence": signal.get("confidence", 0),
                "claimLevel": signal.get("claimLevel", "unknown"),
                "resolvedVia": signal["label"],
                "evidence": signal.get("evidence"),
            })

    limitations = []
    if not any(s.get("provider") == "Google" and s["status"] == "verified" for s in signals):
        limitations.append("Google/Gemini is only verified when a trusted SynthID, manifest payload, or official adapter result is recovered.")
    limitations.append("Classifier attribution is probable only and must not be treated as statutory provenance.")
    limitations.append("The WMAR clustered-token detector requires the experimental Python backend and model/codebook assets.")
    if not primary and has_hints:
        limitations.append("Metadata hints are not statutory provenance on their own.")

    return {
        "verdict": verdict,
        "claimLevel": claim_level,
        "provider": primary.get("provider") if primary else None,
        "providerCandidates": candidates,
        "system": primary.get("system") if primary else None,
        "contentId": primary.get("contentId") if primary else None,
        "createdAt": primary.get("createdAt") if primary else None,
        "signals": signals,
        "confidence": round(float(primary.get("confidence", 0))) if primary else 0,
        "resolvedVia": primary.get("label") if primary else None,
        "officialCheckStatus": _official_check_status(signals, official_results or []),
        "limitations": limitations,
        "record": rec,
    }
