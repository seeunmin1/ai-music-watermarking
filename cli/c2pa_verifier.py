"""C2PA/Content Credentials verification against provider profiles.

When an audio asset path is supplied and c2patool is installed, this module uses
the external contentauth/c2pa-rs CLI as the authority for manifest and trust
validation. Metadata-only matches remain useful evidence, but they do not
produce verified provider provenance without a trusted signature.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any

OFFICIAL_C2PA_TRUST_ANCHORS = (
    "https://raw.githubusercontent.com/c2pa-org/conformance-public/"
    "refs/heads/main/trust-list/C2PA-TRUST-LIST.pem"
)


def _manifest_text(manifest: dict[str, Any], fields: dict[str, Any] | None) -> str:
    return json.dumps({"manifest": manifest, "fields": fields or {}}, default=str).lower()


def _includes_any(text: str, terms: list[str]) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text) for term in terms)


def _first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _run_c2patool(path: str, args: list[str]) -> dict[str, Any]:
    exe = os.getenv("AUDIOMARK_C2PATOOL_PATH") or shutil.which("c2patool")
    if not exe:
        return {
            "available": False,
            "ok": False,
            "status": "tool_not_configured",
            "detail": "c2patool was not found; set AUDIOMARK_C2PATOOL_PATH or add c2patool to PATH.",
        }
    try:
        proc = subprocess.run(
            [exe, path, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(os.getenv("AUDIOMARK_C2PATOOL_TIMEOUT", "30")),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": True,
            "ok": False,
            "status": "tool_error",
            "detail": f"c2patool failed: {exc}",
        }

    return {
        "available": True,
        "ok": proc.returncode == 0,
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "json": _first_json_object(proc.stdout),
    }


def _trust_args() -> list[str]:
    args = ["trust"]
    anchors = os.getenv("AUDIOMARK_C2PA_TRUST_ANCHORS", OFFICIAL_C2PA_TRUST_ANCHORS)
    allowed = os.getenv("AUDIOMARK_C2PA_ALLOWED_LIST")
    trust_config = os.getenv("AUDIOMARK_C2PA_TRUST_CONFIG")
    if anchors:
        args.extend(["--trust_anchors", anchors])
    if allowed:
        args.extend(["--allowed_list", allowed])
    if trust_config:
        args.extend(["--trust_config", trust_config])
    return args


def _validation_items(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    items = report.get("validation_status") or report.get("validationStatus") or []
    out = items if isinstance(items, list) else []
    results = report.get("validation_results") or report.get("validationResults") or {}
    active = results.get("activeManifest") if isinstance(results, dict) else None
    if isinstance(active, dict):
        failures = active.get("failure") or []
        if isinstance(failures, list):
            out = [*out, *failures]
    return out


def _validation_state(report: dict[str, Any] | None) -> str:
    if not isinstance(report, dict):
        return "unknown"
    return str(report.get("validation_state") or report.get("validationState") or "").lower()


def _has_manifest(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    manifest_store = report.get("manifest_store") or report.get("manifestStore")
    manifests = report.get("manifests")
    active = report.get("active_manifest") or report.get("activeManifest")
    return bool(manifest_store or manifests or active or report.get("claim_generator"))


def _extract_fields_from_report(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    active_label = report.get("active_manifest") or report.get("activeManifest")
    manifests = report.get("manifests") if isinstance(report.get("manifests"), dict) else {}
    active = manifests.get(active_label) if active_label in manifests else {}
    assertions = active.get("assertions") if isinstance(active, dict) else None
    generator_info = active.get("claim_generator_info") if isinstance(active, dict) else None
    signature_info = active.get("signature_info") if isinstance(active, dict) else None
    generator_name = None
    if isinstance(generator_info, list) and generator_info:
        first_generator = generator_info[0]
        if isinstance(first_generator, dict):
            generator_name = first_generator.get("name")
    fields: dict[str, Any] = {
        "claim_generator": active.get("claim_generator") or generator_name or report.get("claim_generator") if isinstance(active, dict) else report.get("claim_generator"),
        "title": active.get("title") if isinstance(active, dict) else None,
    }
    if isinstance(signature_info, dict):
        fields["signature_issuer"] = signature_info.get("issuer")
        fields["signature_common_name"] = signature_info.get("common_name")
        fields["signed_at"] = signature_info.get("time")
    if isinstance(assertions, list):
        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue
            data = assertion.get("data") if isinstance(assertion.get("data"), dict) else assertion
            for key in ("provider", "system", "created", "unique_id", "uid", "digitalSourceType"):
                if key in data and data[key] and key not in fields:
                    fields[key] = data[key]
    return {k: v for k, v in fields.items() if v}


def inspect_c2pa_asset(path: str | None, meta: dict[str, Any]) -> dict[str, Any]:
    manifest_report = None
    trust_report = None
    certs_report = None
    tool_status = "not_run"
    tool_detail = "No asset path supplied; using extracted metadata only."

    if path:
        manifest_report = _run_c2patool(path, [])
        trust_report = _run_c2patool(path, _trust_args())
        certs_report = _run_c2patool(path, ["--certs"])
        tool_status = trust_report.get("status", "unknown")
        tool_detail = trust_report.get("detail") or trust_report.get("stderr") or "c2patool trust completed."

    manifest_json = (manifest_report or {}).get("json")
    trust_json = (trust_report or {}).get("json")
    trust_validation_status = _validation_items(trust_json)
    manifest_validation_status = _validation_items(manifest_json)
    validation_status = trust_validation_status if trust_json is not None else manifest_validation_status
    validation_state = _validation_state(trust_json) or _validation_state(manifest_json)
    manifest_found = bool(meta.get("manifest")) or _has_manifest(manifest_json) or _has_manifest(trust_json)
    signature_valid = bool((manifest_report or {}).get("ok") and manifest_found)
    trusted_signature = bool(
        (trust_report or {}).get("ok")
        and manifest_found
        and (validation_state == "trusted" or not trust_validation_status)
    )
    extracted_fields = _extract_fields_from_report(trust_json) or _extract_fields_from_report(manifest_json)
    certificate_text = (certs_report or {}).get("stdout") or ""

    return {
        "manifest_found": manifest_found,
        "signature_valid": signature_valid,
        "trusted_signature": trusted_signature,
        "validation_status": validation_status,
        "validation_state": validation_state,
        "tool_status": tool_status,
        "tool_detail": tool_detail.strip() if isinstance(tool_detail, str) else tool_detail,
        "tool_available": bool((trust_report or {}).get("available")),
        "claim_generator": extracted_fields.get("claim_generator"),
        "certificate_subject": extracted_fields.get("signature_common_name"),
        "certificate_issuer": extracted_fields.get("signature_issuer"),
        "certificate_count": certificate_text.count("BEGIN CERTIFICATE"),
        "fields": extracted_fields,
        "raw_report": trust_json or manifest_json,
    }


def verify_c2pa_manifest(
    meta: dict[str, Any],
    provider_profiles: list[dict[str, Any]],
    asset_path: str | None = None,
    c2pa_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    manifest = meta.get("manifest")
    report = c2pa_report or inspect_c2pa_asset(asset_path, meta)
    if not manifest and not report.get("manifest_found"):
        return []
    fields = {**(meta.get("fields") or {}), **(report.get("fields") or {})}
    text = _manifest_text(manifest or report.get("raw_report") or {}, fields)
    if report.get("claim_generator"):
        text += " " + str(report["claim_generator"]).lower()
    if report.get("certificate_subject"):
        text += " " + str(report["certificate_subject"]).lower()
    if report.get("certificate_issuer"):
        text += " " + str(report["certificate_issuer"]).lower()
    results = []

    for profile in provider_profiles:
        issuers = profile.get("c2pa_issuers", [])
        aliases = profile.get("aliases", [])
        identity = profile.get("c2pa_identity", {})
        identity_terms = [
            *identity.get("certificate_subjects", []),
            *identity.get("certificate_issuers", []),
            *identity.get("claim_generators", []),
        ]
        if not issuers and "c2pa" not in profile.get("signals", []):
            continue
        if not (
            _includes_any(text, issuers)
            or _includes_any(text, identity_terms)
            or _includes_any(str(fields.get("provider", "")), aliases)
        ):
            continue

        policy = profile.get("confidence_policy", {})
        trusted = bool(report.get("trusted_signature"))
        signature_valid = bool(report.get("signature_valid"))
        results.append({
            "id": f"c2pa:{profile['provider_id']}",
            "provider": profile["display_name"],
            "label": f"{profile['display_name']} C2PA Content Credentials",
            "configured": True,
            "verified": trusted,
            "status": "verified" if trusted else "detected",
            "confidence": policy.get("trusted_c2pa", 94) if trusted else policy.get("metadata_hint", 20),
            "system": fields.get("system") or (profile.get("products") or [profile["display_name"]])[0],
            "contentId": fields.get("uid") or fields.get("unique_id"),
            "createdAt": fields.get("created"),
            "detail": (
                "Trusted C2PA signature matched this provider profile."
                if trusted
                else "C2PA manifest matched this provider profile, but no trusted signature was validated."
            ),
            "evidence": {
                "profile": profile["provider_id"],
                "matched": "c2pa_issuer_or_provider_field",
                "manifestFound": bool(report.get("manifest_found")),
                "signatureValid": signature_valid,
                "trustedSignature": trusted,
                "validationStatus": report.get("validation_status", []),
                "validationState": report.get("validation_state"),
                "toolStatus": report.get("tool_status"),
                "toolAvailable": report.get("tool_available"),
                "claimGenerator": report.get("claim_generator"),
            },
        })
    return results
