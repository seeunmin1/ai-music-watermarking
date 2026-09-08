"""Watermark detectors under evaluation.

Two decoders run over the same perturbed audio:

* `detect_baseline` is the shipping decoder. It assumes the payload starts at
  sample 0 and that frames sit on a fixed grid from there.
* `detect_sync_search` is a proposed decoder that finds the payload's actual
  alignment before decoding. It is not part of the product; it exists so the
  robustness report can separate "the watermark was destroyed" from "the
  watermark survived but the decoder looked in the wrong place".

Both return the shipping `Detection` dataclass, so the harness scores them
side by side.

Why the search is exhaustive rather than a coarse sweep: the payload is carried
on a white PN sequence, whose autocorrelation collapses after a single sample of
misalignment. Stepping the offset in strides would miss nearly every true
alignment. So the search covers every sample offset, using an FFT
cross-correlation to get all offsets in one pass, then exploits the payload's
structure to test all alignments as cheap array operations:

  a shift of d samples is q*BIT_LEN + r, so sweeping the sub-bit offset r over
  [0, BIT_LEN) and the payload phase q mod NBITS covers every possible shift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CLI_DIR = Path(__file__).resolve().parents[1] / "cli"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

from audiomark import (  # noqa: E402  (path set up above)
    BIT_LEN,
    FRAME,
    ID_BITS,
    NBITS,
    PREAMBLE,
    Detection,
    checksum24,
    detect_watermark,
    pn_sequence,
)

# Same acceptance threshold the shipping decoder uses.
Z_THRESHOLD = 2.5

_PREAMBLE = np.array(PREAMBLE, dtype=int)


def detect_baseline(samples: np.ndarray, _rate: int | None = None) -> Detection:
    """The decoder as shipped."""
    return detect_watermark(samples)


def _correlate_all_offsets(x: np.ndarray, pn: np.ndarray) -> np.ndarray:
    """c[n] = sum_k x[n+k] * pn[k], for every offset n, via FFT."""
    n_fft = 1 << (len(x) + len(pn) - 1).bit_length()
    spectrum = np.fft.rfft(x, n_fft) * np.conj(np.fft.rfft(pn, n_fft))
    return np.fft.irfft(spectrum, n_fft)[: len(x) - len(pn) + 1]


def _bits_to_ints(bits: np.ndarray) -> np.ndarray:
    """Pack a (n_bits, n_offsets) bit matrix into one integer per column."""
    weights = (1 << np.arange(bits.shape[0] - 1, -1, -1)).astype(np.int64)
    return weights @ bits.astype(np.int64)


def detect_sync_search(samples: np.ndarray, _rate: int | None = None) -> Detection:
    """Search every sample alignment, then decode the best valid payload."""
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < FRAME:
        return Detection(False, 0, 0.0, 0.0, 0)

    pn = pn_sequence()
    correlation = _correlate_all_offsets(x, pn)

    # Column o of `grid` is the bit-correlation series for sub-bit offset o.
    n_bits = len(correlation) // BIT_LEN
    if n_bits < NBITS:
        return Detection(False, 0, 0.0, 0.0, 0)
    grid = correlation[: n_bits * BIT_LEN].reshape(n_bits, BIT_LEN)

    # Fold the repeated payload down: accumulated[r, o] sums every bit whose
    # index is congruent to r modulo NBITS, at offset o.
    n_reps = n_bits // NBITS
    if n_reps < 1:
        return Detection(False, 0, 0.0, 0.0, 0)
    accumulated = grid[: n_reps * NBITS].reshape(n_reps, NBITS, BIT_LEN).sum(axis=0)

    rms = float(np.sqrt(np.mean(x[: n_bits * BIT_LEN] ** 2))) or 1e-6
    sigma = rms * np.sqrt(BIT_LEN * n_reps)

    best = Detection(False, 0, 0.0, 0.0, 0)
    best_z = -1.0

    for phase in range(NBITS):
        # Payload bit j sits at residue (phase + j) mod NBITS.
        payload = np.roll(accumulated, -phase, axis=0)  # (NBITS, BIT_LEN)
        bits = (payload > 0).astype(int)

        z_per_offset = np.mean(np.abs(payload), axis=0) / sigma

        preamble_ok = np.all(bits[: len(PREAMBLE)] == _PREAMBLE[:, None], axis=0)
        record_ids = _bits_to_ints(bits[len(PREAMBLE) : len(PREAMBLE) + ID_BITS])
        checksums = _bits_to_ints(bits[len(PREAMBLE) + ID_BITS :])
        expected = ((record_ids >> 16) ^ (record_ids >> 8) ^ record_ids) & 0xFF
        valid = preamble_ok & (checksums == expected) & (z_per_offset > Z_THRESHOLD)

        # Prefer a fully valid payload; otherwise keep the strongest correlation.
        candidates = np.flatnonzero(valid)
        if candidates.size:
            offset = candidates[np.argmax(z_per_offset[candidates])]
            found = True
        else:
            offset = int(np.argmax(z_per_offset))
            found = False
        z = float(z_per_offset[offset])

        if (found, z) > (best.found, best_z):
            best_z = z
            best = Detection(
                found,
                int(record_ids[offset]),
                z,
                min(99.9, max(0.0, (1 - np.exp(-z / 3)) * 100)),
                n_reps,
            )

    return best


DETECTORS = {
    "baseline": detect_baseline,
    "sync_search": detect_sync_search,
}
