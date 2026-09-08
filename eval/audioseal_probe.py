#!/usr/bin/env python3
"""Benchmark Audiomark's watermark against Meta AudioSeal.

AudioSeal is the useful comparison because it is a *deployed, open* audio
watermark with a public detector - unlike SynthID, whose detector Google does
not release. That makes it the one third-party scheme we can measure against
end to end rather than reason about.

Three questions:

1. **Cross-detection.** Does either tool see the other's mark? If Audiomark
   cannot read an AudioSeal mark, then AudioSeal-watermarked AI audio is
   invisible to us - a coverage gap, since AudioSeal ships in Meta's audio
   models and the provider catalog already lists it.
2. **Perceptibility.** How far under the music does each mark sit? A louder
   watermark buys robustness at the cost of audibility.
3. **Robustness.** Same perturbation battery, same carriers, both decoders.

Everything runs at 16 kHz, which is AudioSeal's operating rate, so neither
scheme is handicapped by resampling the other does not need.

Usage:
  python eval/audioseal_probe.py --duration 40
  python eval/audioseal_probe.py --json eval/results/audioseal.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

EVAL_DIR = Path(__file__).resolve().parent
CLI_DIR = EVAL_DIR.parents[0] / "cli"
for candidate in (str(CLI_DIR), str(EVAL_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import audiomark  # noqa: E402
from dataset import load_items  # noqa: E402
from detectors import detect_baseline, detect_sync_search  # noqa: E402
from perturbations import build_suite  # noqa: E402

AUDIOSEAL_RATE = 16000
TEST_RECORD_ID = 0xA5C3E1

# The perturbations that separated the two decoders, plus controls. Running the
# full battery through a neural detector is slow and adds little.
BENCHMARK_PERTURBATIONS = (
    "none", "mp3_192", "mp3_96", "crop_0s5", "shift_1000",
    "noise_snr20", "lowpass_8k", "compression", "time_stretch_1pct",
)


def load_mono_16k(path: Path, duration_s: float) -> np.ndarray:
    """Decode any supported file to mono float32 at AudioSeal's rate."""
    import miniaudio

    decoded = miniaudio.decode_file(str(path), output_format=miniaudio.SampleFormat.SIGNED16)
    channels = int(getattr(decoded, "nchannels", 1) or 1)
    rate = int(decoded.sample_rate)
    samples = np.asarray(decoded.samples, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    target_len = int(len(samples) * AUDIOSEAL_RATE / rate)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, target_len), np.arange(len(samples)), samples
    )
    return resampled[: int(AUDIOSEAL_RATE * duration_s)].astype(np.float64)


def snr_db(clean: np.ndarray, marked: np.ndarray) -> float:
    length = min(len(clean), len(marked))
    residual = marked[:length] - clean[:length]
    signal = float(np.mean(clean[:length] ** 2))
    noise = float(np.mean(residual**2))
    return 10 * np.log10(signal / noise) if noise > 0 and signal > 0 else float("inf")


class AudioSealWrapper:
    """Thin wrapper so the harness can treat AudioSeal like any other detector."""

    def __init__(self) -> None:
        import torch
        from audioseal import AudioSeal

        self.torch = torch
        self.generator = AudioSeal.load_generator("audioseal_wm_16bits")
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits")

    def _tensor(self, samples: np.ndarray):
        return self.torch.from_numpy(samples.astype(np.float32))[None, None, :]

    def embed(self, samples: np.ndarray) -> np.ndarray:
        wav = self._tensor(samples)
        with self.torch.no_grad():
            watermark = self.generator.get_watermark(wav, AUDIOSEAL_RATE)
        return (wav + watermark).squeeze().numpy().astype(np.float64)

    def detect(self, samples: np.ndarray) -> float:
        """Return the detector's watermark probability."""
        if len(samples) < AUDIOSEAL_RATE:
            return 0.0
        with self.torch.no_grad():
            result, _message = self.detector.detect_watermark(
                self._tensor(samples), AUDIOSEAL_RATE
            )
        return float(np.asarray(result).reshape(-1)[0])


def cross_detection(clean: np.ndarray, seal: AudioSealWrapper) -> dict[str, Any]:
    """Does either scheme detect the other's mark, or a clean file?"""
    audiomark_marked, _ = audiomark.embed_watermark(clean, TEST_RECORD_ID)
    seal_marked = seal.embed(clean)

    def audiomark_verdict(signal: np.ndarray) -> dict[str, Any]:
        detection = detect_sync_search(signal)
        return {
            "found": bool(detection.found),
            "id_correct": bool(detection.found and detection.record_id == TEST_RECORD_ID),
            "z": round(float(detection.z), 2),
        }

    return {
        "clean": {
            "audiomark": audiomark_verdict(clean),
            "audioseal_p": round(seal.detect(clean), 4),
        },
        "audiomark_marked": {
            "audiomark": audiomark_verdict(audiomark_marked),
            "audioseal_p": round(seal.detect(audiomark_marked), 4),
            "snr_db": round(snr_db(clean, audiomark_marked), 1),
        },
        "audioseal_marked": {
            "audiomark": audiomark_verdict(seal_marked),
            "audioseal_p": round(seal.detect(seal_marked), 4),
            "snr_db": round(snr_db(clean, seal_marked), 1),
        },
    }


def robustness_comparison(clean: np.ndarray, seal: AudioSealWrapper) -> list[dict[str, Any]]:
    """Run both marks through the same perturbations at the same rate."""
    audiomark_marked, _ = audiomark.embed_watermark(clean, TEST_RECORD_ID)
    seal_marked = seal.embed(clean)
    suite = {p.name: p for p in build_suite()}

    rows = []
    for name in BENCHMARK_PERTURBATIONS:
        perturbation = suite.get(name)
        if perturbation is None:
            continue
        row: dict[str, Any] = {"perturbation": name, "severity": perturbation.severity}
        try:
            attacked_ours = perturbation.apply(audiomark_marked, AUDIOSEAL_RATE)
            attacked_seal = perturbation.apply(seal_marked, AUDIOSEAL_RATE)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        baseline = detect_baseline(attacked_ours)
        sync = detect_sync_search(attacked_ours)
        row.update({
            "audiomark_baseline": bool(baseline.found and baseline.record_id == TEST_RECORD_ID),
            "audiomark_sync": bool(sync.found and sync.record_id == TEST_RECORD_ID),
            "audiomark_z": round(float(sync.z), 2),
            "audioseal_p": round(seal.detect(attacked_seal), 4),
            "audioseal_survived": seal.detect(attacked_seal) > 0.5,
        })
        rows.append(row)
    return rows


def print_report(report: dict[str, Any]) -> None:
    bar = "=" * 78
    print(bar)
    print("  AUDIOMARK vs META AUDIOSEAL")
    print(bar)
    print(f"  Carrier   : {report['carrier']}")
    print(f"  Rate      : {AUDIOSEAL_RATE} Hz (AudioSeal's operating rate)")
    print(f"  Duration  : {report['duration_s']:.0f} s")

    cross = report["cross_detection"]
    print("\n  CROSS-DETECTION  (can either tool read the other's mark?)")
    print(f"    {'audio under test':<24}{'Audiomark':>14}{'AudioSeal p':>14}")
    print("    " + "-" * 52)
    for key, label in (
        ("clean", "unmarked"),
        ("audiomark_marked", "Audiomark-marked"),
        ("audioseal_marked", "AudioSeal-marked"),
    ):
        entry = cross[key]
        ours = "DETECTED" if entry["audiomark"]["id_correct"] else "not found"
        print(f"    {label:<24}{ours:>14}{entry['audioseal_p']:>14.4f}")

    print("\n  PERCEPTIBILITY  (how far the mark sits under the music)")
    print(f"    Audiomark : {cross['audiomark_marked']['snr_db']:>5.1f} dB SNR")
    print(f"    AudioSeal : {cross['audioseal_marked']['snr_db']:>5.1f} dB SNR")

    print("\n  ROBUSTNESS  (same perturbations, same carrier)")
    print(f"    {'PERTURBATION':<20}{'OURS (shipping)':>17}{'OURS (sync)':>13}{'AUDIOSEAL':>12}")
    print("    " + "-" * 62)
    for row in report["robustness"]:
        if "error" in row:
            print(f"    {row['perturbation']:<20}{'ERROR':>17}")
            continue
        base = "PASS" if row["audiomark_baseline"] else "FAIL"
        sync = "PASS" if row["audiomark_sync"] else "FAIL"
        seal = "PASS" if row["audioseal_survived"] else "FAIL"
        print(f"    {row['perturbation']:<20}{base:>17}{sync:>13}"
              f"{seal:>8} {row['audioseal_p']:.2f}")
    print(bar)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audiomark vs AudioSeal benchmark")
    parser.add_argument("--carrier", type=Path, help="carrier audio file")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    carrier = args.carrier
    if carrier is None:
        candidates = [i.path for i in load_items() if i.path.exists() and i.label == "human"]
        if not candidates:
            candidates = [i.path for i in load_items() if i.path.exists()]
        if not candidates:
            print("No carrier audio found; pass --carrier.", file=sys.stderr)
            return 1
        carrier = candidates[0]

    clean = load_mono_16k(carrier, args.duration)
    if len(clean) < audiomark.FRAME:
        needed = audiomark.FRAME / AUDIOSEAL_RATE
        print(f"Carrier too short: need >= {needed:.1f}s at {AUDIOSEAL_RATE} Hz.", file=sys.stderr)
        return 1

    print("Loading AudioSeal models ...", flush=True)
    seal = AudioSealWrapper()

    report = {
        "carrier": carrier.name,
        "duration_s": len(clean) / AUDIOSEAL_RATE,
        "rate": AUDIOSEAL_RATE,
        "cross_detection": cross_detection(clean, seal),
        "robustness": robustness_comparison(clean, seal),
    }
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
