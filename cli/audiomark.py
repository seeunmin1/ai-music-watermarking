#!/usr/bin/env python3
"""
AUDIOMARK AI — Compliance Engine MVP (Python port)
CA SB 942 / AB 853 statutory watermarking + verification

Same scheme as the web MVP, byte-compatible where it matters:
  • Latent watermark: keyed spread-spectrum PN sequence,
    payload = 12-bit sync preamble + 24-bit record ID + 8-bit checksum,
    repeated across the track, level-adaptive gain.
  • Manifest: C2PA-style JSON written INTO the WAV as a RIFF "c2pa"
    chunk, bound by SHA-256 hashes.
  • Detect: matched-filter correlation (z-score) + byte-level metadata
    scan (embedded C2PA JSON, ID3v2, JUMBF markers, vendor-name hints).
  • Registry + audit log persisted as JSON next to this script.
  • Zero retention: uploaded/scanned audio is never stored.

Only dependency: numpy (stdlib otherwise).

CLI:
  python audiomark.py embed  in.wav out.wav --provider "Audiomark Labs" \
        --system DemoTTS --version 1.0 [--manifest out.json]
  python audiomark.py detect suspicious.wav
  python audiomark.py registry [--revoke RECORD_ID]
  python audiomark.py audit
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import struct
import sys
import uuid
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from c2pa_jumbf import manifest_fields as c2pa_manifest_fields
from c2pa_jumbf import parse_c2pa_store
from local_attribution import DEFAULT_MODEL_PATH, classify, train_centroid_model
from official_adapters import run_official_checks
from provider_catalog import vendor_labels_from_catalog
from provider_discovery import build_report
from provenance_core import VENDOR_LABELS, analyze_provenance
from wmar_experimental import build_eval_command, describe_adapter

# ----------------------------------------------------------------------
# Watermark scheme constants (identical to the JS MVP)
# ----------------------------------------------------------------------
WM_KEY = 0x00A1D107                      # secret embedding key (demo)
BIT_LEN = 2048                           # samples per payload bit
PREAMBLE = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]   # 12-bit sync
ID_BITS = 24
CK_BITS = 8
NBITS = len(PREAMBLE) + ID_BITS + CK_BITS
FRAME = NBITS * BIT_LEN                  # samples per payload repetition

STATE_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = STATE_DIR / "audiomark_registry.json"
AUDIT_PATH = STATE_DIR / "audiomark_audit.json"

VENDOR_HINTS = sorted({*VENDOR_LABELS.keys(), *vendor_labels_from_catalog().keys()})


# ----------------------------------------------------------------------
# Keyed PN sequence — mulberry32, bit-exact with the JS implementation
# so marks embedded by the web MVP decode here and vice versa.
# ----------------------------------------------------------------------
def _mulberry32(seed: int):
    a = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        x = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (x + ((x ^ (x >> 7)) * (61 | x) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t = t ^ x  # note: JS is ((x + imul(...)) ^ x)
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rand


def pn_sequence(length: int = BIT_LEN, key: int = WM_KEY) -> np.ndarray:
    rand = _mulberry32(key)
    return np.array([-1.0 if rand() < 0.5 else 1.0 for _ in range(length)],
                    dtype=np.float64)


def _to_bits(value: int, n: int) -> list[int]:
    return [(value >> i) & 1 for i in range(n - 1, -1, -1)]


def _from_bits(bits) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def checksum24(record_id: int) -> int:
    return ((record_id >> 16) ^ (record_id >> 8) ^ record_id) & 0xFF


# ----------------------------------------------------------------------
# Embed / detect
# ----------------------------------------------------------------------
def embed_watermark(samples: np.ndarray, record_id: int):
    """Embed the payload into mono float samples in [-1, 1].
    Returns (watermarked_samples, repetitions) or (None, 0) if too short."""
    bits = PREAMBLE + _to_bits(record_id, ID_BITS) + _to_bits(checksum24(record_id), CK_BITS)
    signs = np.array([1.0 if b else -1.0 for b in bits])
    pn = pn_sequence()
    out = samples.astype(np.float64).copy()
    if len(out) < FRAME:
        return None, 0

    reps = len(out) // FRAME
    for r in range(reps):
        base = r * FRAME
        for b in range(NBITS):
            s0 = base + b * BIT_LEN
            seg = out[s0:s0 + BIT_LEN]
            rms = float(np.sqrt(np.mean(seg * seg)))
            # level-adaptive gain: ride under local energy, floor for silence
            gain = min(0.12 * rms + 0.0006, 0.02)
            seg += signs[b] * pn * gain
    np.clip(out, -1.0, 1.0, out=out)
    return out, reps


@dataclass
class Detection:
    found: bool
    record_id: int
    z: float
    confidence: float
    repetitions: int


def detect_watermark(samples: np.ndarray) -> Detection:
    """Matched-filter detection with z-score confidence."""
    x = samples.astype(np.float64)
    reps = len(x) // FRAME
    if reps < 1:
        return Detection(False, 0, 0.0, 0.0, 0)

    pn = pn_sequence()
    usable = x[:reps * FRAME].reshape(reps, NBITS, BIT_LEN)
    # correlation of every bit-slot with the PN sequence, summed over reps
    acc = np.einsum("rbs,s->b", usable, pn)          # shape (NBITS,)
    rms = float(np.sqrt(np.mean(usable * usable))) or 1e-6

    bits = (acc > 0).astype(int)
    pre_ok = list(bits[:len(PREAMBLE)]) == PREAMBLE
    rec_id = _from_bits(bits[len(PREAMBLE):len(PREAMBLE) + ID_BITS])
    ck_ok = _from_bits(bits[len(PREAMBLE) + ID_BITS:]) == checksum24(rec_id)

    sigma = rms * np.sqrt(BIT_LEN * reps)            # noise-hypothesis std
    z = float(np.mean(np.abs(acc)) / sigma)
    confidence = min(99.9, max(0.0, (1 - np.exp(-z / 3)) * 100))
    return Detection(pre_ok and ck_ok and z > 2.5, rec_id, z, confidence, reps)


# ----------------------------------------------------------------------
# WAV I/O with a RIFF "c2pa" chunk for the in-file manifest
# ----------------------------------------------------------------------
def read_wav(path: Path):
    """Read a WAV to mono float64 in [-1,1]. Returns (samples, rate, raw_bytes)."""
    raw = path.read_bytes()
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128) / 128.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {width * 8}-bit")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, rate, raw


def _decode_with_miniaudio(path: Path):
    try:
        import miniaudio  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Optional MP3/general audio decoder is not installed. "
            "Install cli requirements to enable MP3 feature classification."
        ) from exc

    decoded = miniaudio.decode_file(
        str(path),
        output_format=miniaudio.SampleFormat.SIGNED16,
    )
    channels = int(getattr(decoded, "nchannels", 1) or 1)
    rate = int(getattr(decoded, "sample_rate", 0) or 0)
    if rate <= 0:
        raise ValueError("Decoded audio did not include a valid sample rate.")
    pcm = np.asarray(decoded.samples, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate, {
        "status": "decoded",
        "codec": path.suffix.lower().lstrip(".") or "unknown",
        "decoder": "miniaudio",
        "sample_rate": rate,
        "channels": channels,
        "duration": len(pcm) / rate,
        "detail": "Audio decoded for watermark detection and local attribution.",
    }


def try_read_audio(path: Path):
    """Best-effort audio reader used by detection.

    WAV files use the stdlib wave path. Other formats use the optional
    miniaudio dependency when available. Metadata and official-provider checks
    still run when waveform decoding is unavailable.
    """
    try:
        samples, rate, raw = read_wav(path)
        return samples, rate, raw, None, {
            "status": "decoded",
            "codec": "wav",
            "decoder": "wave",
            "sample_rate": rate,
            "channels": 1,
            "duration": len(samples) / rate if rate else None,
            "detail": "PCM WAV decoded for watermark detection and local attribution.",
        }
    except (wave.Error, EOFError, ValueError) as exc:
        raw = path.read_bytes()
        wav_error = str(exc)

    try:
        samples, rate, info = _decode_with_miniaudio(path)
        return samples, rate, raw, None, info
    except Exception as exc:
        return None, None, raw, str(exc), {
            "status": "unsupported",
            "codec": path.suffix.lower().lstrip(".") or "unknown",
            "decoder": "miniaudio",
            "sample_rate": None,
            "channels": None,
            "duration": None,
            "detail": f"Waveform decoding skipped: {exc}",
            "wav_error": wav_error,
        }


def write_wav(path: Path, samples: np.ndarray, rate: int,
              extra_chunks: list[tuple[bytes, bytes]] | None = None):
    """Write 16-bit mono WAV; extra_chunks = [(b'c2pa', payload_bytes), ...]."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (np.where(pcm < 0, pcm * 0x8000, pcm * 0x7FFF)).astype("<i2").tobytes()

    chunks = b""
    for cid, data in (extra_chunks or []):
        pad = b"\x00" if len(data) % 2 else b""
        chunks += cid[:4].ljust(4) + struct.pack("<I", len(data)) + data + pad

    body = (b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm16)) + pcm16 + chunks)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


# ----------------------------------------------------------------------
# Metadata scanner (Mode A byte-level pass)
# ----------------------------------------------------------------------
def _extract_json_near(data: bytes, idx: int, window: int = 96,
                       max_len: int = 200_000) -> dict | None:
    start = data.find(b"{", idx, idx + window)
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    end = min(len(data), start + max_len)
    for i in range(start, end):
        c = data[i]
        if esc:
            esc = False
            continue
        if in_str:
            if c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
            continue
        if c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(data[start:i + 1].decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return None
    return None


def scan_metadata(data: bytes) -> dict:
    """Find embedded C2PA JSON, ID3v2/JUMBF markers, and vendor hints."""
    low = data.lower()

    def _is_token_byte(value: int) -> bool:
        return (48 <= value <= 57) or (97 <= value <= 122)

    def _vendor_present(v: str) -> bool:
        needle = v.encode()
        i = low.find(needle)
        while i >= 0:
            # "udio" is a substring of "audio" — require a non-'a' predecessor
            before_ok = i == 0 or not _is_token_byte(low[i - 1])
            after = i + len(needle)
            after_ok = after >= len(low) or not _is_token_byte(low[after])
            if before_ok and after_ok:
                return True
            i = low.find(needle, i + 1)
        return False

    res = {"id3": data[:3] == b"ID3",
           "jumbf": b"jumb" in low or b"jumd" in low,
           "manifest": None, "fields": None, "manifest_format": None,
           "vendor_hints": [v for v in VENDOR_HINTS if _vendor_present(v)]}

    # Production C2PA is CBOR inside JUMBF boxes, which is what Google, Adobe
    # and other providers actually ship. Try that first; the JSON scan below
    # only recognizes the manifests this project writes into its own WAVs.
    jumbf_manifest = parse_c2pa_store(data)
    if jumbf_manifest:
        res["manifest"] = jumbf_manifest
        res["fields"] = c2pa_manifest_fields(jumbf_manifest)
        res["manifest_format"] = "jumbf_cbor"
        return res

    i = 0
    while res["manifest"] is None:
        idx = low.find(b"c2pa", i)
        if idx < 0:
            break
        res["manifest"] = _extract_json_near(data, idx)
        i = idx + 4

    if res["manifest"]:
        res["manifest_format"] = "audiomark_json"
        m = res["manifest"]
        stat = next((a.get("data", {}) for a in m.get("assertions", [])
                     if "statutory" in a.get("label", "")), {})
        res["fields"] = {
            "provider": stat.get("provider") or m.get("claim_generator", "Unknown"),
            "system": (f"{stat['system']} v{stat.get('system_version', '?')}"
                       if stat.get("system") else m.get("claim_generator", "—")),
            "created": stat.get("created"),
            "unique_id": stat.get("unique_id"),
            "uid": stat.get("unique_id"),
        }
    return res


# ----------------------------------------------------------------------
# Manifest, registry, audit
# ----------------------------------------------------------------------
def build_manifest(rec: dict) -> dict:
    return {
        "@context": "https://c2pa.org/manifest (illustrative)",
        "claim_generator": "AudiomarkAI/0.3.0-py",
        "title": rec["file_name"],
        "assertions": [
            {"label": "c2pa.actions",
             "data": {"actions": [{"action": "c2pa.created",
                                   "digitalSourceType": "trainedAlgorithmicMedia"}]}},
            {"label": "ai.statutory_disclosure.ca_sb942",
             "data": {"provider": rec["provider"], "system": rec["system"],
                      "system_version": rec["version"], "created": rec["timestamp"],
                      "unique_id": rec["uid"]}},
            {"label": "audiomark.latent_watermark",
             "data": {"scheme": "audiomark-ss/v1", "record_id": rec["id_hex"],
                      "bit_length": NBITS, "redundancy": rec["repetitions"]}},
        ],
        "content_bindings": {"alg": "sha256", "source_asset": rec["source_hash"]},
        "signature": {"alg": "ed25519 (simulated in MVP)",
                      "value": "am1_" + rec["uid"].replace("-", "")[:40]},
    }


def _load(path: Path) -> list:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return []


def _save(path: Path, value: list):
    path.write_text(json.dumps(value, indent=2))


def audit_log(event_type: str, detail: str):
    log = _load(AUDIT_PATH)
    log.insert(0, {"t": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                   "type": event_type, "detail": detail})
    _save(AUDIT_PATH, log[:500])


# ----------------------------------------------------------------------
# High-level operations
# ----------------------------------------------------------------------
def op_embed(in_path: Path, out_path: Path, provider: str, system: str,
             version: str, manifest_out: Path | None = None) -> dict:
    samples, rate, raw = read_wav(in_path)
    if len(samples) < FRAME:
        sys.exit(f"error: audio must be ≥ {FRAME / rate:.1f} s at {rate} Hz "
                 f"to carry one payload frame")

    record_id = int.from_bytes(hashlib.sha256(uuid.uuid4().bytes).digest()[:3],
                               "big")  # random 24-bit ID
    marked, reps = embed_watermark(samples, record_id)

    rec = {
        "id_hex": f"{record_id:06x}",
        "provider": provider, "system": system, "version": version,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "uid": str(uuid.uuid4()),
        "file_name": in_path.name,
        "source_hash": hashlib.sha256(raw).hexdigest(),
        "repetitions": reps, "status": "active",
    }
    manifest = build_manifest(rec)
    man_bytes = json.dumps(manifest).encode()
    write_wav(out_path, marked, rate, [(b"c2pa", man_bytes)])
    rec["wav_hash"] = hashlib.sha256(out_path.read_bytes()).hexdigest()

    registry = _load(REGISTRY_PATH)
    registry.insert(0, rec)
    _save(REGISTRY_PATH, registry)
    audit_log("embed", f"Embedded record {rec['id_hex']} + in-file C2PA manifest "
                       f"into '{in_path.name}' ({reps}x redundancy) for "
                       f"{provider}/{system} v{version}")
    if manifest_out:
        manifest_out.write_text(json.dumps(manifest, indent=2))
    return rec


def op_detect(path: Path) -> dict:
    samples, rate, raw, decode_error, audio_decode = try_read_audio(path)
    det = detect_watermark(samples) if samples is not None else Detection(False, 0, 0.0, 0.0, 0)
    meta = scan_metadata(raw)

    registry = _load(REGISTRY_PATH)
    official_results = run_official_checks(str(path), meta)
    classifier_result = (
        classify(samples, rate, DEFAULT_MODEL_PATH)
        if samples is not None and rate is not None
        else {
            "status": "unsupported",
            "candidates": [],
            "detail": f"Local audio classifier skipped because this file could not be decoded: {decode_error}",
        }
    )
    prov = analyze_provenance(det, meta, registry, official_results, classifier_result)
    if samples is None:
        prov["limitations"].append(
            "Latent Audiomark watermark detection and local attribution require decoded waveform audio; this upload still ran metadata, C2PA, and official-provider checks."
        )
    ai_detected = prov["verdict"] == "ai_generated"

    audit_log("verify",
              (f"AI-GENERATED verdict for '{path.name}' - "
               f"resolved via {prov['resolvedVia']}")
              if ai_detected else
              f"{'UNKNOWN WITH HINTS' if prov['verdict'] == 'unknown_with_hints' else 'UNMARKED'} "
              f"verdict for '{path.name}' (z={det.z:.1f}, no verified payload)")
    return {"detection": det, "metadata": meta, "provenance": prov,
            "ai_detected": ai_detected, "audio_decode": audio_decode,
            "classifier": classifier_result}

    rec = next((r for r in registry
                if int(r["id_hex"], 16) == det.record_id), None) if det.found else None

    ai_detected = det.found or meta["manifest"] is not None
    if rec:
        prov = {"provider": rec["provider"],
                "system": f"{rec['system']} v{rec['version']}",
                "created": rec["timestamp"], "unique_id": rec["uid"],
                "source": "Registry (latent watermark)",
                "record_status": rec["status"]}
    elif meta["fields"]:
        prov = {**meta["fields"], "source": "Embedded C2PA manifest"}
    elif det.found:
        prov = {"provider": "Unknown (unregistered mark)", "system": "—",
                "created": None, "unique_id": f"record 0x{det.record_id:06x}",
                "source": "Latent watermark (no registry match)"}
    else:
        prov = None

    audit_log("verify",
              (f"AI-GENERATED verdict for '{path.name}' — "
               f"{'watermark z=%.1f' % det.z if det.found else 'no watermark'}"
               f"{', C2PA manifest found' if meta['manifest'] else ''}")
              if ai_detected else
              f"UNMARKED verdict for '{path.name}' (z={det.z:.1f}, no manifest)")
    return {"detection": det, "metadata": meta, "provenance": prov,
            "ai_detected": ai_detected}


def op_revoke(record_id_hex: str):
    registry = _load(REGISTRY_PATH)
    for r in registry:
        if r["id_hex"] == record_id_hex and r["status"] == "active":
            r["status"] = "revocation_pending"
            r["revoked_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
            deadline = (_dt.datetime.now(_dt.timezone.utc)
                        + _dt.timedelta(hours=96)).isoformat()
            r["revocation_deadline"] = deadline
            _save(REGISTRY_PATH, registry)
            audit_log("revocation",
                      f"Strip/violation flagged on record {record_id_hex} "
                      f"('{r['file_name']}'). 96-hour revocation window opened "
                      f"— deadline {deadline}")
            return r
    sys.exit(f"error: no active record {record_id_hex}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _print_detect(res: dict, name: str):
    det: Detection = res["detection"]
    meta = res["metadata"]
    prov = res["provenance"]
    bar = "-" * 62
    verdict = "AI-GENERATED" if res["ai_detected"] else (
        ("AI-GENERATED (DECLARED) / SIGNATURE NOT VALIDATED"
         if prov.get("declaresAiGenerated")
         else "C2PA MANIFEST FOUND / SIGNER NOT TRUSTED")
        if prov["generationClaimLevel"] == "detected"
        else "PROBABLY AI-GENERATED" if prov["claimLevel"] == "probable"
        else "UNKNOWN WITH METADATA HINTS" if prov["verdict"] == "unknown_with_hints"
        else "HUMAN CREATED / UNMARKED"
    )
    print(bar)
    print(f"  STATUTORY VERDICT: {verdict}")
    print(bar)
    print(f"  Latent watermark : "
          f"{'DECODED (%dx reps)' % det.repetitions if det.found else 'not detected'}"
          f"   z={det.z:.2f}  confidence={det.confidence:.1f}%")
    print(f"  C2PA manifest    : {'PARSED' if meta['manifest'] else 'not found'}")
    print(f"  JUMBF markers    : {'present' if meta['jumbf'] else 'not found'}")
    print(f"  ID3v2 block      : {'present' if meta['id3'] else 'not found'}")
    if meta["vendor_hints"]:
        hints = ", ".join(VENDOR_LABELS.get(v, v) for v in meta["vendor_hints"])
        print(f"  Vendor strings   : {hints}")
    google_verified = any(s["id"] == "google_synthid" and s["status"] == "verified"
                          for s in prov["signals"])
    print(f"  SynthID decoder  : {'VERIFIED PAYLOAD' if google_verified else 'unsupported official decoder'}")
    print("  WMAR detector    : experimental adapter hook")
    print(bar)
    if prov["provider"]:
        print("  PROVIDER IDENTIFICATION" if prov["providerVerificationStatus"] == "verified" else "  PROVIDER ATTRIBUTION")
        print(f"    Generation claim    : {prov['generationClaimLevel']}")
        print(f"    Provider claim      : {prov['providerClaimLevel']}")
        print(f"    Provider verified   : {prov['providerVerificationStatus']}")
        print(f"    Provider / Platform : {prov['provider']}")
        print(f"    System version      : {prov['system'] or '-'}")
        print(f"    Timestamp           : {prov.get('createdAt') or '-'}")
        print(f"    Unique content ID   : {prov.get('contentId') or '-'}")
        print(f"    Resolved via        : {prov.get('resolvedVia') or '-'}")
        if (prov.get("record") or {}).get("status") == "revocation_pending":
            print("    record is inside its 96-hour revocation window")
    else:
        print("  No statutory payload recovered. Absence of a mark is consistent")
        print("  with human-created content but cannot rule out stripped or")
        print("  never-marked AI output.")
    if prov["signals"]:
        print("  SIGNALS")
        for signal in prov["signals"]:
            print(f"    {signal['category']:<22} {signal['label']}: {signal['status']}")
    if prov["limitations"]:
        print("  LIMITATIONS")
        for limitation in prov["limitations"]:
            print(f"    - {limitation}")
    print(bar)
    print(f"  '{name}' was analyzed in memory - nothing retained.")
    return
    bar = "─" * 62
    print(bar)
    verdict = "AI-GENERATED" if res["ai_detected"] else "HUMAN CREATED / UNMARKED"
    print(f"  STATUTORY VERDICT: {verdict}")
    print(bar)
    print(f"  Latent watermark : "
          f"{'DECODED (%dx reps)' % det.repetitions if det.found else 'not detected'}"
          f"   z={det.z:.2f}  confidence={det.confidence:.1f}%")
    print(f"  C2PA manifest    : {'PARSED' if meta['manifest'] else 'not found'}")
    print(f"  JUMBF markers    : {'present' if meta['jumbf'] else 'not found'}")
    print(f"  ID3v2 block      : {'present' if meta['id3'] else 'not found'}")
    if meta["vendor_hints"]:
        hints = ", ".join(VENDOR_LABELS.get(v, v) for v in meta["vendor_hints"])
        print(f"  Vendor strings   : {hints}")
    print(f"  SynthID decoder  : STUB · integration pending")
    print(f"  AudioSeal decoder: STUB · integration pending")
    print(bar)
    prov = res["provenance"]
    if prov:
        print("  PROVENANCE IDENTIFICATION")
        print(f"    Provider / Platform : {prov['provider']}")
        print(f"    System version      : {prov['system']}")
        print(f"    Timestamp           : {prov.get('created') or '—'}")
        print(f"    Unique content ID   : {prov.get('unique_id') or '—'}")
        print(f"    Resolved via        : {prov['source']}")
        if prov.get("record_status") == "revocation_pending":
            print("    ⚠ record is inside its 96-hour revocation window")
    else:
        print("  No statutory payload recovered. Absence of a mark is consistent")
        print("  with human-created content but cannot rule out stripped or")
        print("  never-marked AI output.")
    print(bar)
    print(f"  '{name}' was analyzed in memory — nothing retained.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Audiomark AI compliance engine (MVP)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="Mode B: embed watermark + in-file manifest")
    pe.add_argument("input", type=Path)
    pe.add_argument("output", type=Path)
    pe.add_argument("--provider", default="Audiomark Labs")
    pe.add_argument("--system", default="DemoTTS")
    pe.add_argument("--version", default="1.0")
    pe.add_argument("--manifest", type=Path, help="also write manifest JSON here")

    pd = sub.add_parser("detect", help="Mode A: scan a file, print statutory verdict")
    pd.add_argument("input", type=Path)
    pd.add_argument("--json", action="store_true", help="print normalized result JSON")

    wmar = sub.add_parser("wmar-info", help="describe the experimental nograd-audio-wm adapter")
    wmar.add_argument("--model-family", choices=["musicgen_encodec", "moshi_mimi", "cosyvoice", "sparktts"],
                      help="print a reproducible upstream clustered-token eval command")

    train = sub.add_parser("train-attribution", help="train the local probable-attribution model from labeled WAV folders")
    train.add_argument("dataset_dir", type=Path)
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)

    discover = sub.add_parser("provider-discovery", help="report provider catalog coverage and adapter status")
    discover.add_argument("--output", type=Path)

    pr = sub.add_parser("registry", help="list records / flag a strip")
    pr.add_argument("--revoke", metavar="RECORD_ID",
                    help="flag strip → open 96-hour revocation window")

    sub.add_parser("audit", help="print the audit log")

    a = p.parse_args(argv)

    if a.cmd == "embed":
        rec = op_embed(a.input, a.output, a.provider, a.system, a.version, a.manifest)
        print(f"REGISTERED {rec['id_hex']}  →  {a.output}")
        print(f"  provider {rec['provider']} / {rec['system']} v{rec['version']}")
        print(f"  unique ID {rec['uid']}")
        print(f"  {rec['repetitions']}x payload redundancy · manifest embedded in-file")

    elif a.cmd == "detect":
        res = op_detect(a.input)
        if a.json:
            print(json.dumps({
                "detection": asdict(res["detection"]),
                "metadata": {k: v for k, v in res["metadata"].items() if k != "manifest"},
                "provenance": res["provenance"],
                "ai_detected": res["ai_detected"],
                "audio_decode": res["audio_decode"],
                "classifier": res["classifier"],
            }, indent=2))
        else:
            _print_detect(res, a.input.name)

    elif a.cmd == "wmar-info":
        print(json.dumps(build_eval_command(a.model_family) if a.model_family else describe_adapter(), indent=2))

    elif a.cmd == "train-attribution":
        model = train_centroid_model(a.dataset_dir, a.output)
        for provider, rec in model["providers"].items():
            print(f"{provider}: {rec['samples']} samples")
        print(f"wrote {a.output}")

    elif a.cmd == "provider-discovery":
        report = build_report()
        payload = json.dumps(report, indent=2)
        if a.output:
            a.output.write_text(payload, encoding="utf-8")
        print(payload)

    elif a.cmd == "registry":
        if a.revoke:
            r = op_revoke(a.revoke)
            print(f"record {r['id_hex']} → revocation_pending "
                  f"(deadline {r['revocation_deadline']})")
        else:
            regs = _load(REGISTRY_PATH)
            if not regs:
                print("registry empty — run `embed` first")
            for r in regs:
                print(f"{r['id_hex']}  {r['status']:<19} {r['provider']} / "
                      f"{r['system']} v{r['version']}  {r['file_name']}  "
                      f"{r['timestamp']}")

    elif a.cmd == "audit":
        for e in _load(AUDIT_PATH):
            print(f"{e['t']}  [{e['type'].upper():<10}] {e['detail']}")


if __name__ == "__main__":
    main()
