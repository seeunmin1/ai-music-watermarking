"""Producer-style audio perturbations for watermark robustness testing.

The meeting question was "how do we survive producer audio perturbations?".
These are the transforms a track actually passes through between generation and
release: lossy delivery codecs, gain staging, EQ, sample-rate conversion,
compression/limiting, layering, and edits that move the timeline.

Every transform takes mono float64 samples in [-1, 1] plus a sample rate and
returns the same shape of data (except the edit transforms, which intentionally
change length). Only numpy is required; MP3 round-trips additionally use
`lameenc` to encode and `miniaudio` to decode, so the test exercises a real
codec rather than a simulation of one.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

Transform = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class Perturbation:
    """One named attack, with the severity metadata the report groups by."""

    name: str
    category: str
    severity: str  # "mild" | "moderate" | "severe"
    apply: Transform
    detail: str = ""
    params: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Filter and resampling primitives
# ----------------------------------------------------------------------
def _biquad(samples: np.ndarray, b: tuple[float, float, float], a: tuple[float, float, float]) -> np.ndarray:
    """Direct-form-I biquad. Coefficients follow the RBJ audio cookbook."""
    b0, b1, b2 = (c / a[0] for c in b)
    a1, a2 = a[1] / a[0], a[2] / a[0]
    out = np.empty_like(samples)
    x1 = x2 = y1 = y2 = 0.0
    for i, x0 in enumerate(samples):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return out


def _lowpass(samples: np.ndarray, rate: int, cutoff: float, q: float = 0.707) -> np.ndarray:
    w0 = 2 * math.pi * min(cutoff, rate / 2 - 1) / rate
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b = ((1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2)
    a = (1 + alpha, -2 * cos_w0, 1 - alpha)
    return _biquad(samples, b, a)


def _highpass(samples: np.ndarray, rate: int, cutoff: float, q: float = 0.707) -> np.ndarray:
    w0 = 2 * math.pi * max(cutoff, 1.0) / rate
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b = ((1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2)
    a = (1 + alpha, -2 * cos_w0, 1 - alpha)
    return _biquad(samples, b, a)


def _resample_linear(samples: np.ndarray, ratio: float) -> np.ndarray:
    """Linear-interpolation resampler, modelling a cheap sample-rate converter."""
    if ratio == 1.0 or len(samples) == 0:
        return samples.copy()
    target_len = max(1, int(round(len(samples) * ratio)))
    source_idx = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(source_idx, np.arange(len(samples)), samples)


def _round_trip_resample(samples: np.ndarray, rate: int, intermediate_rate: int) -> np.ndarray:
    down = _resample_linear(samples, intermediate_rate / rate)
    back = _resample_linear(down, rate / intermediate_rate)
    return _fit_length(back, len(samples))


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) == length:
        return samples
    if len(samples) > length:
        return samples[:length]
    return np.pad(samples, (0, length - len(samples)))


# ----------------------------------------------------------------------
# Codec round-trip
# ----------------------------------------------------------------------
# MPEG-1 (32/44.1/48 kHz) and MPEG-2 LSF (16/22.05/24 kHz) permit different
# bitrate ladders. Asking for a bitrate the layer cannot carry makes LAME switch
# layers and resample the output, which rescales the timeline - that reads as a
# destroyed watermark when the codec did nothing of the sort.
_MPEG1_BITRATES = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_MPEG2_BITRATES = (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)


def legal_bitrate(rate: int, requested: int) -> int:
    """Largest permitted bitrate at `rate` that does not exceed `requested`."""
    ladder = _MPEG1_BITRATES if rate >= 32000 else _MPEG2_BITRATES
    allowed = [b for b in ladder if b <= requested]
    return allowed[-1] if allowed else ladder[0]


def mp3_round_trip(samples: np.ndarray, rate: int, bitrate: int = 192) -> np.ndarray:
    """Encode to MP3 at `bitrate` kbps and decode back, as a release would.

    The bitrate is clamped to one the sample rate's MPEG layer supports, so the
    round trip tests codec transparency rather than an accidental resample.
    """
    import lameenc
    import miniaudio

    bitrate = legal_bitrate(rate, bitrate)

    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (np.where(pcm16 < 0, pcm16 * 0x8000, pcm16 * 0x7FFF)).astype("<i2")

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(rate)
    # LAME picks its own output rate otherwise: at 16 kHz in it emits 44.1 kHz,
    # which rescales the timeline and looks like a destroyed watermark rather
    # than the codec transparency this perturbation is meant to test.
    encoder.set_out_sample_rate(rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    # Keep the encoder from writing an ID3/Info header where the build supports
    # it; older lameenc builds do not expose the call and simply include one.
    silence = getattr(encoder, "silence", None)
    if callable(silence):
        try:
            silence()
        except TypeError:
            pass
    payload = encoder.encode(pcm16.tobytes()) + encoder.flush()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "round_trip.mp3"
        path.write_bytes(bytes(payload))
        decoded = miniaudio.decode_file(str(path), output_format=miniaudio.SampleFormat.SIGNED16)

    out = np.asarray(decoded.samples, dtype=np.int16).astype(np.float64) / 32768.0
    channels = int(getattr(decoded, "nchannels", 1) or 1)
    if channels > 1:
        out = out.reshape(-1, channels).mean(axis=1)

    # lameenc honors neither set_out_sample_rate nor low input rates: below
    # 32 kHz it emits 44.1 kHz regardless. Convert back so the perturbation
    # measures codec loss and its delay, not an unrequested rate change.
    decoded_rate = int(getattr(decoded, "sample_rate", rate) or rate)
    if decoded_rate != rate:
        out = _resample_linear(out, rate / decoded_rate)
    return out


# ----------------------------------------------------------------------
# Individual transforms
# ----------------------------------------------------------------------
def add_noise(samples: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    signal_power = float(np.mean(samples**2))
    if signal_power <= 0:
        return samples.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(seed)
    return samples + rng.normal(0.0, math.sqrt(noise_power), len(samples))


def apply_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    return samples * (10 ** (gain_db / 20))


def peak_normalize(samples: np.ndarray, target: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) or 1e-9
    return samples * (target / peak)


def quantize(samples: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** (bits - 1)
    return np.round(np.clip(samples, -1.0, 1.0) * levels) / levels


def compress_dynamics(samples: np.ndarray, rate: int, threshold_db: float = -20.0, ratio: float = 4.0) -> np.ndarray:
    """Simple RMS-envelope compressor with 10 ms attack / 100 ms release."""
    threshold = 10 ** (threshold_db / 20)
    attack = math.exp(-1.0 / (0.010 * rate))
    release = math.exp(-1.0 / (0.100 * rate))
    envelope = 0.0
    out = np.empty_like(samples)
    for i, x in enumerate(samples):
        level = abs(float(x))
        coeff = attack if level > envelope else release
        envelope = coeff * envelope + (1 - coeff) * level
        if envelope > threshold:
            gain = (threshold + (envelope - threshold) / ratio) / (envelope or 1e-9)
        else:
            gain = 1.0
        out[i] = x * gain
    return out


def soft_clip(samples: np.ndarray, drive: float = 1.6) -> np.ndarray:
    return np.tanh(samples * drive) / math.tanh(drive)


def crop_head(samples: np.ndarray, rate: int, seconds: float) -> np.ndarray:
    offset = int(rate * seconds)
    return samples[offset:] if offset < len(samples) else samples.copy()


def shift_samples(samples: np.ndarray, offset: int) -> np.ndarray:
    """Delay the track by `offset` samples, as an edit or codec delay would."""
    return np.concatenate([np.zeros(offset), samples])


def mix_background(samples: np.ndarray, rate: int, level_db: float = -12.0, seed: int = 7) -> np.ndarray:
    """Layer filtered noise under the track, standing in for an added stem."""
    rng = np.random.default_rng(seed)
    bed = rng.normal(0.0, 1.0, len(samples))
    bed = _lowpass(bed, rate, 2000.0)
    bed_rms = float(np.sqrt(np.mean(bed**2))) or 1e-9
    signal_rms = float(np.sqrt(np.mean(samples**2))) or 1e-9
    bed = bed * (signal_rms * (10 ** (level_db / 20)) / bed_rms)
    return samples + bed


def time_stretch_resample(samples: np.ndarray, factor: float) -> np.ndarray:
    """Speed/pitch change by resampling, e.g. a 1% tempo nudge."""
    return _resample_linear(samples, 1.0 / factor)


def add_echo(samples: np.ndarray, rate: int, delay_ms: float = 80.0, decay: float = 0.35) -> np.ndarray:
    delay = int(rate * delay_ms / 1000)
    out = samples.copy()
    if delay < len(samples):
        out[delay:] += decay * samples[:-delay]
    return out


def fade_edges(samples: np.ndarray, rate: int, seconds: float = 1.0) -> np.ndarray:
    n = min(int(rate * seconds), len(samples) // 2)
    if n <= 0:
        return samples.copy()
    out = samples.copy()
    ramp = np.linspace(0.0, 1.0, n)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


# ----------------------------------------------------------------------
# The registered suite
# ----------------------------------------------------------------------
def build_suite() -> list[Perturbation]:
    """The perturbation battery, ordered roughly by how common it is in release."""
    return [
        Perturbation("none", "control", "mild", lambda s, r: s.copy(),
                     "Unmodified watermarked audio (control)"),

        # Lossy delivery — every released track goes through at least one of these.
        Perturbation("mp3_320", "codec", "mild", lambda s, r: mp3_round_trip(s, r, 320),
                     "MP3 encode/decode at 320 kbps", {"bitrate": 320}),
        Perturbation("mp3_192", "codec", "moderate", lambda s, r: mp3_round_trip(s, r, 192),
                     "MP3 encode/decode at 192 kbps", {"bitrate": 192}),
        Perturbation("mp3_128", "codec", "moderate", lambda s, r: mp3_round_trip(s, r, 128),
                     "MP3 encode/decode at 128 kbps", {"bitrate": 128}),
        Perturbation("mp3_96", "codec", "severe", lambda s, r: mp3_round_trip(s, r, 96),
                     "MP3 encode/decode at 96 kbps", {"bitrate": 96}),

        # Gain staging and mastering.
        Perturbation("gain_plus_6db", "gain", "mild", lambda s, r: apply_gain(s, 6.0),
                     "+6 dB gain", {"gain_db": 6.0}),
        Perturbation("gain_minus_6db", "gain", "mild", lambda s, r: apply_gain(s, -6.0),
                     "-6 dB gain", {"gain_db": -6.0}),
        Perturbation("peak_normalize", "gain", "mild", lambda s, r: peak_normalize(s),
                     "Peak normalization to -0.2 dBFS"),
        Perturbation("compression", "dynamics", "moderate", lambda s, r: compress_dynamics(s, r),
                     "4:1 compression above -20 dBFS"),
        Perturbation("soft_clip", "dynamics", "moderate", lambda s, r: soft_clip(s),
                     "Soft-clip limiting / saturation"),

        # EQ.
        Perturbation("lowpass_16k", "eq", "mild", lambda s, r: _lowpass(s, r, 16000.0),
                     "16 kHz low-pass"),
        Perturbation("lowpass_8k", "eq", "severe", lambda s, r: _lowpass(s, r, 8000.0),
                     "8 kHz low-pass"),
        Perturbation("highpass_100", "eq", "mild", lambda s, r: _highpass(s, r, 100.0),
                     "100 Hz high-pass"),

        # Noise and bit depth.
        Perturbation("noise_snr40", "noise", "mild", lambda s, r: add_noise(s, 40.0),
                     "Additive noise at 40 dB SNR", {"snr_db": 40}),
        Perturbation("noise_snr30", "noise", "moderate", lambda s, r: add_noise(s, 30.0),
                     "Additive noise at 30 dB SNR", {"snr_db": 30}),
        Perturbation("noise_snr20", "noise", "severe", lambda s, r: add_noise(s, 20.0),
                     "Additive noise at 20 dB SNR", {"snr_db": 20}),
        Perturbation("quantize_8bit", "quantization", "severe", lambda s, r: quantize(s, 8),
                     "Requantization to 8-bit"),

        # Sample-rate conversion.
        Perturbation("resample_22k", "resample", "moderate", lambda s, r: _round_trip_resample(s, r, 22050),
                     "Round-trip through 22.05 kHz"),
        Perturbation("resample_48k", "resample", "mild", lambda s, r: _round_trip_resample(s, r, 48000),
                     "Round-trip through 48 kHz"),

        # Timeline edits — these break frame alignment, not just signal quality.
        Perturbation("crop_0s5", "edit", "moderate", lambda s, r: crop_head(s, r, 0.5),
                     "Drop the first 0.5 s", {"seconds": 0.5}),
        Perturbation("shift_1000", "edit", "moderate", lambda s, r: shift_samples(s, 1000),
                     "Insert 1000 samples of lead-in", {"offset": 1000}),
        Perturbation("time_stretch_1pct", "edit", "severe", lambda s, r: time_stretch_resample(s, 1.01),
                     "1% tempo/pitch change", {"factor": 1.01}),
        Perturbation("fade_edges", "edit", "mild", lambda s, r: fade_edges(s, r),
                     "1 s fade in and out"),

        # Layering and space.
        Perturbation("mix_bed_minus12db", "layering", "moderate", lambda s, r: mix_background(s, r, -12.0),
                     "Layer a background stem at -12 dB", {"level_db": -12.0}),
        Perturbation("echo", "layering", "moderate", lambda s, r: add_echo(s, r),
                     "80 ms echo at 35% decay"),

        # Realistic release chain: master, then deliver lossy.
        Perturbation("chain_master_mp3", "chain", "severe",
                     lambda s, r: mp3_round_trip(peak_normalize(compress_dynamics(s, r)), r, 192),
                     "Compression + normalization + 192 kbps MP3"),
    ]
