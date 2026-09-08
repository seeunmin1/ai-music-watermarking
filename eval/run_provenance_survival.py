#!/usr/bin/env python3
"""Does provenance survive the trip from generation to a listener?

The full-corpus run showed 4.8% recall on AI music, against 100% on tracks
downloaded straight from the generator. This harness isolates why.

C2PA manifests, ID3 frames and vendor strings all live in the file *container*,
alongside the audio rather than inside it. Re-encoding rebuilds the container.
So the question is not whether a provider disclosed at generation time, but
whether the disclosure is still there after the file has been through the
processing every distribution path applies.

For each file that carries a manifest, this re-encodes it the way a platform
would on upload and re-runs metadata detection on the result. A watermark
embedded in the samples is unaffected by any of this - which is the point.

Usage:
  python eval/run_provenance_survival.py
  python eval/run_provenance_survival.py --json eval/results/survival.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
CLI_DIR = EVAL_DIR.parents[0] / "cli"
for candidate in (str(CLI_DIR), str(EVAL_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from dataset import load_items  # noqa: E402

import audiomark  # noqa: E402


def _decode(path: Path) -> tuple[np.ndarray, int, int]:
    import miniaudio

    decoded = miniaudio.decode_file(str(path), output_format=miniaudio.SampleFormat.SIGNED16)
    return (
        np.asarray(decoded.samples, dtype=np.int16),
        int(decoded.sample_rate),
        int(getattr(decoded, "nchannels", 1) or 1),
    )


def transcode_mp3(path: Path, bitrate: int) -> bytes:
    """Re-encode to MP3, as an upload pipeline or a DAW bounce would."""
    import lameenc

    from perturbations import legal_bitrate

    samples, rate, channels = _decode(path)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(legal_bitrate(rate, bitrate))
    encoder.set_in_sample_rate(rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)
    return bytes(encoder.encode(samples.tobytes()) + encoder.flush())


def rewrite_wav(path: Path) -> bytes:
    """Bounce to PCM WAV, as an editor or converter would."""
    samples, rate, channels = _decode(path)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "bounce.wav"
        audiomark.write_wav(target, samples.astype(np.float64) / 32768.0, rate)
        return target.read_bytes()


def strip_container_tags(path: Path) -> bytes:
    """Drop a leading ID3 tag, the cheapest possible metadata strip."""
    data = path.read_bytes()
    if data[:3] != b"ID3":
        return data
    size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
    return data[size + 10 :]


OPERATIONS: dict[str, tuple[str, Callable[[Path], bytes]]] = {
    "original": ("As downloaded from the generator", lambda p: p.read_bytes()),
    "mp3_320": ("Re-encoded to 320 kbps MP3", lambda p: transcode_mp3(p, 320)),
    "mp3_192": ("Re-encoded to 192 kbps MP3", lambda p: transcode_mp3(p, 192)),
    "mp3_128": ("Re-encoded to 128 kbps MP3", lambda p: transcode_mp3(p, 128)),
    "wav_bounce": ("Bounced to PCM WAV", rewrite_wav),
    "id3_stripped": ("ID3 tag removed", strip_container_tags),
}


def probe(data: bytes) -> dict[str, Any]:
    meta = audiomark.scan_metadata(data)
    manifest = meta.get("manifest") or {}
    return {
        "manifest_found": bool(meta.get("manifest")),
        "manifest_format": meta.get("manifest_format"),
        "declares_ai": bool(isinstance(manifest, dict) and manifest.get("declares_ai_generated")),
        "jumbf": bool(meta.get("jumbf")),
        "id3": bool(meta.get("id3")),
        "vendor_hints": meta.get("vendor_hints", []),
        "bytes": len(data),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provenance survival through re-encoding")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--files", nargs="*", type=Path, help="specific files to test")
    args = parser.parse_args(argv)

    if args.files:
        candidates = [p for p in args.files if p.exists()]
    else:
        # Only files that actually carry provenance can lose it.
        candidates = [
            item.path for item in load_items()
            if item.path.exists() and audiomark.scan_metadata(item.path.read_bytes()).get("manifest")
        ]

    if not candidates:
        print("No files with a provenance manifest found.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for path in candidates:
        for name, (description, operation) in OPERATIONS.items():
            try:
                result = probe(operation(path))
                result.update({"file": path.name, "operation": name, "description": description})
            except Exception as exc:
                result = {"file": path.name, "operation": name, "description": description,
                          "error": f"{type(exc).__name__}: {exc}"}
            rows.append(result)

    bar = "=" * 84
    print(bar)
    print("  PROVENANCE SURVIVAL THROUGH RE-ENCODING")
    print(bar)
    print(f"  Files with a manifest: {len(candidates)}")
    print("\n  Share of files still carrying a readable AI-generation claim:\n")
    print(f"    {'OPERATION':<16}{'SURVIVED':>12}{'RATE':>9}   {'WHAT IT MODELS':<38}")
    print("    " + "-" * 76)

    summary = {}
    for name, (description, _) in OPERATIONS.items():
        subset = [r for r in rows if r["operation"] == name and "error" not in r]
        survived = sum(1 for r in subset if r["declares_ai"])
        rate = survived / len(subset) if subset else 0.0
        summary[name] = {"survived": survived, "total": len(subset), "rate": round(rate, 4)}
        flag = "" if rate == 1.0 else ("  <-- lost" if rate == 0.0 else "  <-- partial")
        print(f"    {name:<16}{f'{survived}/{len(subset)}':>12}{rate:>9.0%}   {description:<38}{flag}")

    print("\n  A watermark lives in the samples, so none of these operations touch it.")
    print("  Container metadata is rebuilt by every one of them.")
    print(bar)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                             encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
