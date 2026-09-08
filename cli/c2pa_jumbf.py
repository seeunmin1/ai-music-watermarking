"""Native JUMBF/CBOR reader for real-world C2PA manifest stores.

`scan_metadata` originally recognized only the JSON manifest this project writes
into its own RIFF `c2pa` chunk. Production C2PA (the Content Credentials that
Google, Adobe, OpenAI and others actually ship) is CBOR encoded inside JUMBF
boxes, carried in an ID3 `GEOB` frame for MP3, a `C2PA` chunk for WAV, or a
`uuid` box for MP4. Those files parsed as "no manifest", so a signed
"Created by Google Generative AI" disclosure was reported as a metadata hint.

This module walks the JUMBF box tree and decodes the CBOR payloads with no
third-party dependency, so detection works whether or not `c2patool` is
installed on the host. It deliberately does **not** validate the signature: the
certificate chain is parsed for identity only. Cryptographic trust remains the
job of `c2pa_verifier`/`c2patool`, so anything found here stays at claim level
"detected" until a trusted signature is validated.
"""

from __future__ import annotations

import struct
from typing import Any

# JUMBF box type codes we care about (ISO/IEC 19566-5).
_BOX_SUPER = b"jumb"
_BOX_DESCRIPTION = b"jumd"
_BOX_CBOR = b"cbor"
_BOX_JSON = b"json"
_BOX_BINARY = b"bfdb"
_BOX_EMBEDDED = b"bidb"

# Largest store we will walk, as a guard against malformed length fields.
_MAX_STORE_BYTES = 64 * 1024 * 1024
_MAX_DEPTH = 16

# IPTC digital source type indicating synthetic media.
_AI_SOURCE_TYPES = (
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
)


class CborError(ValueError):
    """Raised when a CBOR payload cannot be decoded."""


# ----------------------------------------------------------------------
# Minimal CBOR decoder (RFC 8949)
# ----------------------------------------------------------------------
class _Undefined:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "undefined"


UNDEFINED = _Undefined()


def _decode_head(data: bytes, pos: int) -> tuple[int, int, int]:
    """Return (major_type, argument, next_pos). argument is -1 for indefinite."""
    if pos >= len(data):
        raise CborError("truncated CBOR head")
    initial = data[pos]
    major = initial >> 5
    minor = initial & 0x1F
    pos += 1
    if minor < 24:
        return major, minor, pos
    if minor == 24:
        if pos + 1 > len(data):
            raise CborError("truncated uint8 argument")
        return major, data[pos], pos + 1
    if minor == 25:
        if pos + 2 > len(data):
            raise CborError("truncated uint16 argument")
        return major, struct.unpack_from(">H", data, pos)[0], pos + 2
    if minor == 26:
        if pos + 4 > len(data):
            raise CborError("truncated uint32 argument")
        return major, struct.unpack_from(">I", data, pos)[0], pos + 4
    if minor == 27:
        if pos + 8 > len(data):
            raise CborError("truncated uint64 argument")
        return major, struct.unpack_from(">Q", data, pos)[0], pos + 8
    if minor == 31:
        return major, -1, pos
    raise CborError(f"reserved CBOR additional information {minor}")


def _decode_item(data: bytes, pos: int, depth: int = 0) -> tuple[Any, int]:
    if depth > 64:
        raise CborError("CBOR nesting too deep")
    if pos >= len(data):
        raise CborError("truncated CBOR item")
    # For major type 7 the additional information selects the value's kind,
    # while `arg` carries the raw payload bits, so both are needed below.
    minor = data[pos] & 0x1F
    major, arg, pos = _decode_head(data, pos)

    if major == 0:  # unsigned integer
        return arg, pos
    if major == 1:  # negative integer
        return -1 - arg, pos

    if major in (2, 3):  # byte string / text string
        if arg == -1:  # indefinite length: concatenate chunks until break
            chunks = []
            while True:
                if pos < len(data) and data[pos] == 0xFF:
                    pos += 1
                    break
                chunk, pos = _decode_item(data, pos, depth + 1)
                chunks.append(chunk if isinstance(chunk, bytes) else str(chunk).encode())
            joined = b"".join(chunks)
            return joined if major == 2 else joined.decode("utf-8", "replace"), pos
        end = pos + arg
        if end > len(data):
            raise CborError("truncated CBOR string")
        raw = data[pos:end]
        return (raw if major == 2 else raw.decode("utf-8", "replace")), end

    if major == 4:  # array
        out: list[Any] = []
        if arg == -1:
            while True:
                if pos >= len(data):
                    raise CborError("unterminated indefinite array")
                if data[pos] == 0xFF:
                    pos += 1
                    break
                item, pos = _decode_item(data, pos, depth + 1)
                out.append(item)
            return out, pos
        for _ in range(arg):
            item, pos = _decode_item(data, pos, depth + 1)
            out.append(item)
        return out, pos

    if major == 5:  # map
        result: dict[Any, Any] = {}
        if arg == -1:
            while True:
                if pos >= len(data):
                    raise CborError("unterminated indefinite map")
                if data[pos] == 0xFF:
                    pos += 1
                    break
                key, pos = _decode_item(data, pos, depth + 1)
                value, pos = _decode_item(data, pos, depth + 1)
                result[key if isinstance(key, (str, int, bytes)) else str(key)] = value
            return result, pos
        for _ in range(arg):
            key, pos = _decode_item(data, pos, depth + 1)
            value, pos = _decode_item(data, pos, depth + 1)
            result[key if isinstance(key, (str, int, bytes)) else str(key)] = value
        return result, pos

    if major == 6:  # tagged value - keep the payload, drop the tag
        value, pos = _decode_item(data, pos, depth + 1)
        return value, pos

    # major == 7: simple values and floats, selected by the additional
    # information; `arg` holds the raw IEEE-754 bits for the float cases.
    if minor == 20:
        return False, pos
    if minor == 21:
        return True, pos
    if minor == 22:
        return None, pos
    if minor == 23:
        return UNDEFINED, pos
    if minor == 25:
        return struct.unpack(">e", struct.pack(">H", arg))[0], pos
    if minor == 26:
        return struct.unpack(">f", struct.pack(">I", arg))[0], pos
    if minor == 27:
        return struct.unpack(">d", struct.pack(">Q", arg))[0], pos
    return None, pos


def decode_cbor(data: bytes) -> Any:
    """Decode a single CBOR item, ignoring any trailing bytes."""
    value, _ = _decode_item(data, 0)
    return value


# ----------------------------------------------------------------------
# JUMBF box tree
# ----------------------------------------------------------------------
def _parse_description(payload: bytes) -> dict[str, Any]:
    """Decode a `jumd` box: 16-byte type UUID, toggles, optional label."""
    if len(payload) < 17:
        return {"type_uuid": b"", "label": None}
    type_uuid = payload[:16]
    toggles = payload[16]
    pos = 17
    label = None
    if toggles & 0x02:  # a label is present, NUL terminated
        end = payload.find(b"\x00", pos)
        if end < 0:
            end = len(payload)
        label = payload[pos:end].decode("utf-8", "replace")
        pos = end + 1
    return {
        "type_uuid": type_uuid,
        # The first four bytes of the type UUID are the human-readable box type.
        "box_type": type_uuid[:4].decode("ascii", "replace") if type_uuid else "",
        "label": label,
        "toggles": toggles,
    }


def _walk(data: bytes, start: int, end: int, depth: int = 0) -> list[dict[str, Any]]:
    """Walk sibling boxes in [start, end) and return a parsed tree."""
    boxes: list[dict[str, Any]] = []
    pos = start
    while pos + 8 <= end and depth <= _MAX_DEPTH:
        length = int.from_bytes(data[pos : pos + 4], "big")
        box_type = data[pos + 4 : pos + 8]
        if length == 0:  # extends to the end of the parent
            length = end - pos
        elif length == 1:  # 64-bit extended length
            if pos + 16 > end:
                break
            length = int.from_bytes(data[pos + 8 : pos + 16], "big")
            if length < 16 or pos + length > end:
                break
            boxes.append({"type": box_type, "start": pos + 16, "end": pos + length})
            pos += length
            continue
        if length < 8 or pos + length > end:
            break

        body_start, body_end = pos + 8, pos + length
        node: dict[str, Any] = {"type": box_type, "start": body_start, "end": body_end}

        if box_type == _BOX_SUPER:
            node["children"] = _walk(data, body_start, body_end, depth + 1)
        elif box_type == _BOX_DESCRIPTION:
            node.update(_parse_description(data[body_start:body_end]))
        boxes.append(node)
        pos += length
    return boxes


def find_jumbf_stores(data: bytes) -> list[tuple[int, int]]:
    """Locate top-level JUMBF superboxes as (start, end) byte ranges."""
    candidates: list[tuple[int, int]] = []
    search = 0
    while True:
        idx = data.find(_BOX_SUPER, search)
        if idx < 0:
            break
        search = idx + 4
        box_start = idx - 4
        if box_start < 0:
            continue
        length = int.from_bytes(data[box_start : box_start + 4], "big")
        if length < 16 or length > _MAX_STORE_BYTES or box_start + length > len(data):
            continue
        # A superbox always begins with a description box.
        if data[box_start + 8 : box_start + 12] != b"\x00\x00\x00\x00" and (
            data[box_start + 12 : box_start + 16] != _BOX_DESCRIPTION
        ):
            inner_len = int.from_bytes(data[box_start + 8 : box_start + 12], "big")
            if data[box_start + 12 : box_start + 16] != _BOX_DESCRIPTION or inner_len < 8:
                continue
        candidates.append((box_start, box_start + length))

    # Keep only outermost ranges so nested superboxes are not reported twice.
    candidates.sort(key=lambda r: (r[0], -(r[1] - r[0])))
    top: list[tuple[int, int]] = []
    for start, end in candidates:
        if top and start < top[-1][1]:
            continue
        top.append((start, end))
    return top


# ----------------------------------------------------------------------
# Certificate identity (parsed for naming only, never for trust)
# ----------------------------------------------------------------------
_OID_COMMON_NAME = b"\x06\x03\x55\x04\x03"
_OID_ORGANIZATION = b"\x06\x03\x55\x04\x0a"


def _der_names(data: bytes, oid: bytes) -> list[str]:
    """Pull X.509 name strings following a given attribute-type OID."""
    out: list[str] = []
    search = 0
    while True:
        idx = data.find(oid, search)
        if idx < 0:
            break
        search = idx + len(oid)
        pos = idx + len(oid)
        if pos + 2 > len(data):
            break
        tag, length = data[pos], data[pos + 1]
        # PrintableString / UTF8String / IA5String, short form lengths only.
        if tag in (0x0C, 0x13, 0x16) and 0 < length < 0x80 and pos + 2 + length <= len(data):
            value = data[pos + 2 : pos + 2 + length].decode("utf-8", "replace").strip()
            if value and value not in out:
                out.append(value)
    return out


def certificate_identity(signature_bytes: bytes) -> dict[str, list[str]]:
    """Best-effort issuer/subject names from the COSE signature's cert chain."""
    return {
        "common_names": _der_names(signature_bytes, _OID_COMMON_NAME),
        "organizations": _der_names(signature_bytes, _OID_ORGANIZATION),
    }


# ----------------------------------------------------------------------
# C2PA manifest extraction
# ----------------------------------------------------------------------
def _collect(data: bytes, nodes: list[dict[str, Any]], out: dict[str, Any], label_path: str = "") -> None:
    """Recursively gather labeled CBOR/JSON payloads from a JUMBF tree."""
    description = next((n for n in nodes if n["type"] == _BOX_DESCRIPTION), None)
    label = (description or {}).get("label") or ""
    path = f"{label_path}/{label}" if label else label_path
    if label:
        out["labels"].append(path)

    for node in nodes:
        box_type = node["type"]
        payload = data[node["start"] : node["end"]]

        if box_type == _BOX_SUPER:
            _collect(data, node.get("children", []), out, path)
        elif box_type == _BOX_CBOR:
            try:
                value = decode_cbor(payload)
            except (CborError, struct.error, UnicodeDecodeError):
                continue
            out["payloads"].append({"label": path, "kind": "cbor", "data": value})
            if label.startswith("c2pa.signature") or path.endswith("c2pa.signature"):
                out["signature_bytes"] = payload
        elif box_type == _BOX_JSON:
            import json as _json

            try:
                out["payloads"].append(
                    {"label": path, "kind": "json", "data": _json.loads(payload.decode("utf-8"))}
                )
            except (ValueError, UnicodeDecodeError):
                continue
        elif box_type in (_BOX_BINARY, _BOX_EMBEDDED) and label.startswith("c2pa.signature"):
            out["signature_bytes"] = payload


def _walk_values(value: Any):
    """Yield every dict nested anywhere inside a decoded CBOR structure."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)


def parse_c2pa_store(data: bytes) -> dict[str, Any] | None:
    """Parse a real C2PA manifest store out of raw file bytes.

    Returns a normalized manifest dict, or None when no JUMBF C2PA store is
    present. The result intentionally reports *claims*, not verified trust.
    """
    for start, end in find_jumbf_stores(data):
        tree = _walk(data, start + 8, end)
        if not tree:
            continue
        collected: dict[str, Any] = {"labels": [], "payloads": [], "signature_bytes": b""}
        _collect(data, tree, collected)
        if not collected["payloads"] and not collected["labels"]:
            continue
        if not any("c2pa" in label for label in collected["labels"]):
            continue

        actions: list[dict[str, Any]] = []
        source_types: list[str] = []
        claim_generator = None
        instance_id = None
        assertions: list[dict[str, Any]] = []

        for payload in collected["payloads"]:
            label = payload["label"]
            value = payload["data"]
            if isinstance(value, dict) or isinstance(value, list):
                assertions.append({"label": label.rsplit("/", 1)[-1], "data": value})
            for node in _walk_values(value):
                generator_info = node.get("claim_generator_info")
                if isinstance(generator_info, dict) and generator_info.get("name"):
                    claim_generator = claim_generator or generator_info["name"]
                elif isinstance(generator_info, list):
                    for entry in generator_info:
                        if isinstance(entry, dict) and entry.get("name"):
                            claim_generator = claim_generator or entry["name"]
                            break
                if not claim_generator and node.get("claim_generator"):
                    claim_generator = node["claim_generator"]
                if not instance_id and node.get("instanceID"):
                    instance_id = node["instanceID"]
                if node.get("action"):
                    actions.append(node)
                source_type = node.get("digitalSourceType")
                if isinstance(source_type, str) and source_type not in source_types:
                    source_types.append(source_type)

        identity = certificate_identity(collected["signature_bytes"]) if collected["signature_bytes"] else {
            "common_names": [],
            "organizations": [],
        }
        # Fall back to scanning the whole store when the signature box was not
        # isolated cleanly; certificates live inside it either way.
        if not identity["organizations"]:
            identity = certificate_identity(data[start:end])

        ai_generated = any(
            any(marker in source_type.lower() for marker in _AI_SOURCE_TYPES)
            for source_type in source_types
        )

        return {
            "format": "jumbf_cbor",
            "claim_generator": claim_generator,
            "instance_id": instance_id,
            "actions": actions,
            "assertions": assertions,
            "labels": collected["labels"],
            "digital_source_types": source_types,
            "declares_ai_generated": ai_generated,
            "signature_present": bool(collected["signature_bytes"]),
            "certificate_common_names": identity["common_names"],
            "certificate_organizations": identity["organizations"],
        }
    return None


def manifest_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    """Map a parsed JUMBF manifest onto the flat field shape detection expects."""
    organizations = manifest.get("certificate_organizations") or []
    common_names = manifest.get("certificate_common_names") or []
    provider = organizations[0] if organizations else (manifest.get("claim_generator") or "Unknown")
    descriptions = [
        action.get("description")
        for action in manifest.get("actions", [])
        if isinstance(action, dict) and action.get("description")
    ]
    return {
        "provider": provider,
        "system": manifest.get("claim_generator") or (common_names[0] if common_names else "Unknown"),
        "created": None,
        "unique_id": manifest.get("instance_id"),
        "uid": manifest.get("instance_id"),
        "digitalSourceType": (manifest.get("digital_source_types") or [None])[0],
        "descriptions": descriptions,
    }
