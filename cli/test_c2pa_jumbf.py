"""Tests for the native JUMBF/CBOR C2PA reader."""

from __future__ import annotations

import struct

import pytest

from c2pa_jumbf import (
    CborError,
    certificate_identity,
    decode_cbor,
    find_jumbf_stores,
    manifest_fields,
    parse_c2pa_store,
)
from audiomark import scan_metadata

TRAINED_ALGORITHMIC = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


# ----------------------------------------------------------------------
# Minimal CBOR encoder, used only to build fixtures
# ----------------------------------------------------------------------
def _head(major: int, argument: int) -> bytes:
    if argument < 24:
        return bytes([major << 5 | argument])
    if argument < 0x100:
        return bytes([major << 5 | 24, argument])
    if argument < 0x10000:
        return bytes([major << 5 | 25]) + struct.pack(">H", argument)
    if argument < 0x100000000:
        return bytes([major << 5 | 26]) + struct.pack(">I", argument)
    return bytes([major << 5 | 27]) + struct.pack(">Q", argument)


def cbor(value) -> bytes:
    if value is True:
        return b"\xf5"
    if value is False:
        return b"\xf4"
    if value is None:
        return b"\xf6"
    if isinstance(value, int):
        return _head(0, value) if value >= 0 else _head(1, -1 - value)
    if isinstance(value, bytes):
        return _head(2, len(value)) + value
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return _head(3, len(encoded)) + encoded
    if isinstance(value, list):
        return _head(4, len(value)) + b"".join(cbor(v) for v in value)
    if isinstance(value, dict):
        return _head(5, len(value)) + b"".join(cbor(k) + cbor(v) for k, v in value.items())
    raise TypeError(f"unsupported fixture type {type(value)}")


# ----------------------------------------------------------------------
# JUMBF fixture builders
# ----------------------------------------------------------------------
def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + box_type + payload


def _description(type_code: bytes, label: str) -> bytes:
    uuid = type_code.ljust(4, b" ") + b"\x00\x11\x00\x10\x80\x00\x00\xaa\x008\x9bq\x03"
    return _box(b"jumd", uuid[:16] + b"\x03" + label.encode() + b"\x00")


def _superbox(type_code: bytes, label: str, *children: bytes) -> bytes:
    return _box(b"jumb", _description(type_code, label) + b"".join(children))


def build_store(actions: list[dict], signature: bytes = b"") -> bytes:
    """Assemble a manifest store shaped like the ones providers actually ship."""
    assertion = _superbox(
        b"cbor", "c2pa.actions.v2", _box(b"cbor", cbor({"actions": actions}))
    )
    assertions = _superbox(b"c2as", "c2pa.assertions", assertion)
    claim = _superbox(
        b"cbor",
        "c2pa.claim.v2",
        _box(b"cbor", cbor({
            "instanceID": "1234abcd-0000-1111-2222-333344445555",
            "claim_generator_info": {"name": "Test Generator Library", "version": "1.0"},
        })),
    )
    signature_box = _superbox(b"c2cs", "c2pa.signature", _box(b"cbor", signature or cbor({})))
    manifest = _superbox(b"c2ma", "urn:c2pa:test-manifest", claim, assertions, signature_box)
    return _superbox(b"c2pa", "c2pa", manifest)


AI_ACTIONS = [
    {"action": "c2pa.created",
     "description": "Created by Test Generative AI.",
     "digitalSourceType": TRAINED_ALGORITHMIC},
]
HUMAN_ACTIONS = [
    {"action": "c2pa.created",
     "description": "Captured with a camera.",
     "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"},
]


# ----------------------------------------------------------------------
# CBOR decoding
# ----------------------------------------------------------------------
@pytest.mark.parametrize("value", [
    0, 1, 23, 24, 255, 256, 65535, 65536, 4294967296,
    -1, -24, -256, -70000,
    b"", b"\x00\x01\x02",
    "", "hello", "unicode ✓ ok",
    [], [1, 2, 3], [[1], [2, [3]]],
    {}, {"a": 1}, {"nested": {"deep": [1, {"x": True}]}},
    True, False, None,
])
def test_cbor_round_trip(value):
    assert decode_cbor(cbor(value)) == value


def test_cbor_decodes_indefinite_length_containers():
    # 0x9f ... 0xff is an indefinite-length array; 0xbf ... 0xff a map.
    assert decode_cbor(b"\x9f\x01\x02\x03\xff") == [1, 2, 3]
    assert decode_cbor(b"\xbf\x61a\x01\xff") == {"a": 1}


def test_cbor_decodes_tagged_value_as_its_payload():
    tagged = b"\xd2" + cbor([1, 2])  # tag 18 (COSE_Sign1) wrapping an array
    assert decode_cbor(tagged) == [1, 2]


def test_cbor_decodes_floats():
    assert decode_cbor(b"\xfb\x40\x09\x21\xfb\x54\x44\x2d\x18") == pytest.approx(3.14159265)
    assert decode_cbor(b"\xfa\x40\x49\x0f\xdb") == pytest.approx(3.14159, rel=1e-5)


def test_cbor_raises_on_truncated_input():
    with pytest.raises(CborError):
        decode_cbor(b"\x19\x01")  # uint16 header with only one byte of argument


def test_cbor_raises_on_reserved_additional_information():
    with pytest.raises(CborError):
        decode_cbor(b"\x1c")


# ----------------------------------------------------------------------
# Store discovery and parsing
# ----------------------------------------------------------------------
def test_finds_store_embedded_in_surrounding_bytes():
    store = build_store(AI_ACTIONS)
    blob = b"ID3\x03\x00\x00\x00\x00/2GEOB\x00" + store + b"\xff\xfb" + b"\x00" * 512
    assert find_jumbf_stores(blob)
    assert parse_c2pa_store(blob) is not None


def test_parses_ai_generation_claim():
    manifest = parse_c2pa_store(build_store(AI_ACTIONS))
    assert manifest is not None
    assert manifest["declares_ai_generated"] is True
    assert manifest["claim_generator"] == "Test Generator Library"
    assert manifest["instance_id"] == "1234abcd-0000-1111-2222-333344445555"
    assert TRAINED_ALGORITHMIC in manifest["digital_source_types"]
    assert any(a.get("action") == "c2pa.created" for a in manifest["actions"])


def test_camera_capture_is_not_flagged_as_ai_generated():
    manifest = parse_c2pa_store(build_store(HUMAN_ACTIONS))
    assert manifest is not None
    assert manifest["declares_ai_generated"] is False


def test_returns_none_without_a_store():
    assert parse_c2pa_store(b"\x00" * 4096) is None
    assert parse_c2pa_store(b"ID3\x03\x00\x00just an ordinary mp3 tag") is None


def test_ignores_malformed_length_fields():
    # A 'jumb' marker whose length field runs past the buffer must not parse.
    blob = b"\xff\xff\xff\xffjumb" + b"\x00" * 64
    assert parse_c2pa_store(blob) is None


def test_manifest_fields_maps_onto_detection_shape():
    manifest = parse_c2pa_store(build_store(AI_ACTIONS))
    fields = manifest_fields(manifest)
    assert fields["unique_id"] == "1234abcd-0000-1111-2222-333344445555"
    assert fields["digitalSourceType"] == TRAINED_ALGORITHMIC
    assert "Created by Test Generative AI." in fields["descriptions"]


def test_certificate_identity_reads_x509_names():
    # SEQUENCE-free fixture: the OID for CN followed by a PrintableString.
    der = b"\x06\x03\x55\x04\x03\x13\x0bExample Ltd"
    assert certificate_identity(der)["common_names"] == ["Example Ltd"]


def test_certificate_identity_on_empty_input():
    assert certificate_identity(b"") == {"common_names": [], "organizations": []}


# ----------------------------------------------------------------------
# Integration with the metadata scanner
# ----------------------------------------------------------------------
def test_scan_metadata_reports_jumbf_manifest():
    blob = b"ID3\x03\x00\x00\x00\x00/2GEOB\x00" + build_store(AI_ACTIONS)
    meta = scan_metadata(blob)
    assert meta["manifest_format"] == "jumbf_cbor"
    assert meta["manifest"]["declares_ai_generated"] is True
    assert meta["fields"]["unique_id"] == "1234abcd-0000-1111-2222-333344445555"


def test_scan_metadata_still_reads_audiomark_json_manifests():
    # The project's own manifests are JSON in a RIFF chunk, not JUMBF.
    payload = (
        b'{"claim_generator":"AudiomarkAI/0.3.0-py","assertions":['
        b'{"label":"ai.statutory_disclosure.ca_sb942","data":{"provider":"Audiomark Labs",'
        b'"system":"DemoTTS","system_version":"1.0","unique_id":"abc-123"}}]}'
    )
    blob = b"RIFF\x00\x00\x00\x00WAVEc2pa" + struct.pack("<I", len(payload)) + payload
    meta = scan_metadata(blob)
    assert meta["manifest_format"] == "audiomark_json"
    assert meta["fields"]["provider"] == "Audiomark Labs"


def test_scan_metadata_without_any_manifest():
    meta = scan_metadata(b"\x00" * 2048)
    assert meta["manifest"] is None
    assert meta["manifest_format"] is None
