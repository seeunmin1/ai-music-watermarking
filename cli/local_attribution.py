"""Trainable local audio-provider attribution model.

This classifier is intentionally conservative. It outputs probable attribution
only when a saved centroid model exists and the top score clears the threshold.
It never produces verified provenance.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "attribution_model.json"
MIN_CONFIDENCE = 0.68


def _read_wav_features(path: Path) -> tuple[np.ndarray, int]:
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
    return data, rate


def extract_features(samples: np.ndarray, sample_rate: int) -> list[float]:
    x = samples.astype(np.float64)
    if len(x) == 0:
        return [0.0] * 10
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    spec = np.abs(np.fft.rfft(x[: min(len(x), sample_rate * 8)] * np.hanning(min(len(x), sample_rate * 8))))
    freqs = np.fft.rfftfreq((len(spec) - 1) * 2, 1 / sample_rate) if len(spec) > 1 else np.array([0.0])
    total = float(np.sum(spec)) or 1e-9
    centroid = float(np.sum(freqs * spec) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / total))
    rolloff = float(freqs[min(len(freqs) - 1, int(np.searchsorted(np.cumsum(spec), total * 0.85)))])
    low = float(np.sum(spec[freqs < 300]) / total)
    mid = float(np.sum(spec[(freqs >= 300) & (freqs < 3000)]) / total)
    high = float(np.sum(spec[freqs >= 3000]) / total)
    duration = float(len(x) / sample_rate)
    return [duration, rms, peak, zcr, centroid, bandwidth, rolloff, low, mid, high]


def extract_file_features(path: Path) -> list[float]:
    samples, rate = _read_wav_features(path)
    return extract_features(samples, rate)


def train_centroid_model(dataset_dir: Path, output_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    providers = {}
    for provider_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        vectors = []
        for wav_path in sorted(provider_dir.glob("*.wav")):
            vectors.append(extract_file_features(wav_path))
        if vectors:
            arr = np.array(vectors, dtype=np.float64)
            providers[provider_dir.name] = {
                "centroid": arr.mean(axis=0).tolist(),
                "scale": (arr.std(axis=0) + 1e-6).tolist(),
                "samples": len(vectors),
            }
    model = {
        "version": 1,
        "type": "centroid_audio_attribution",
        "min_confidence": MIN_CONFIDENCE,
        "providers": providers,
        "feature_names": ["duration", "rms", "peak", "zcr", "centroid", "bandwidth", "rolloff", "low", "mid", "high"],
    }
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def classify(samples: np.ndarray, sample_rate: int, model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        return {
            "status": "not_configured",
            "candidates": [],
            "detail": f"No local attribution model found at {model_path}.",
        }
    model = json.loads(model_path.read_text(encoding="utf-8"))
    features = np.array(extract_features(samples, sample_rate), dtype=np.float64)
    scored = []
    for provider, rec in model.get("providers", {}).items():
        centroid = np.array(rec["centroid"], dtype=np.float64)
        scale = np.array(rec["scale"], dtype=np.float64)
        dist = float(np.sqrt(np.mean(((features - centroid) / scale) ** 2)))
        score = 1.0 / (1.0 + math.exp(dist - 2.0))
        scored.append({"provider": provider, "confidence": score, "distance": dist})
    scored.sort(key=lambda x: x["confidence"], reverse=True)
    threshold = float(model.get("min_confidence", MIN_CONFIDENCE))
    if not scored or scored[0]["confidence"] < threshold:
        return {
            "status": "abstain",
            "candidates": scored[:3],
            "detail": "Local attribution classifier abstained below confidence threshold.",
        }
    return {
        "status": "probable",
        "candidates": scored[:3],
        "detail": "Local audio attribution classifier matched nearest calibrated provider profile.",
    }
