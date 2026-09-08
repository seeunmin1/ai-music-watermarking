# MVP validation report

Prepared for the Tuesday review. Covers what the evaluation harnesses measure,
what they found, and what is still unmeasured.

```bash
.venv/bin/python eval/fetch_corpus.py --suno 40 --udio 40 --human 40
.venv/bin/python eval/run_detection_eval.py
.venv/bin/python eval/run_provenance_survival.py
.venv/bin/python eval/run_robustness.py --duration 30
.venv/bin/python eval/audioseal_probe.py --duration 40
```

---

## The headline

On a 126-file corpus spanning three generators, detection recall is **4.8%** —
4 of 84 AI tracks. Not because the code is wrong: precision and specificity are
both **100%**, and the four it catches it identifies perfectly.

**It is because provenance metadata does not survive redistribution.** A single
re-encode destroys it, and 95% of the AI corpus had already been through one.

That reframes the product. Reading C2PA is necessary and now works, but it only
ever sees files fetched straight from the generator. Everything downstream needs
a watermark.

---

## 1. Detection missed real C2PA manifests — fixed

The four Gemini tracks each carry a **fully signed Google C2PA manifest**:

```
c2pa.created  "Created by Google Generative AI."
              digitalSourceType: …/trainedAlgorithmicMedia
c2pa.edited   "Applied imperceptible SynthID watermark."

signed by   Google LLC / Google Media Processing Services
chaining to Google C2PA Root CA G3
validation  Trusted  (official C2PA trust list)
```

Before the fix the tool called these **“Unknown with metadata hints”** — the
strongest possible disclosure signal, on the exact statutory use case, scored as
a near-miss.

**Cause.** `scan_metadata` searched for the string `c2pa` and parsed what
followed as **JSON**. Production C2PA is **CBOR inside JUMBF boxes**, in an ID3
`GEOB` frame. The tool understood only the manifests it writes itself.

**Fix.** `cli/c2pa_jumbf.py`, a dependency-free JUMBF walker and CBOR decoder.
It reads identity only; trust stays `c2patool`'s job, so an unvalidated manifest
stays at claim level *detected*, never *verified*.

## 2. Detection accuracy at corpus scale

| Cohort | Source | n |
| --- | --- | --- |
| Gemini | direct download | 4 |
| Suno | SONICS (ICLR 2025) | 40 |
| Udio | SONICS (ICLR 2025) | 40 |
| Human | FMA small, 8 genres | 40 + 1 |
| AI-assisted | Timothy's game track | 1 |

All three verdict policies score identically, because nothing lands in the
intermediate claim levels — a file either has a readable manifest or nothing:

| Metric | Value |
| --- | --- |
| Accuracy | **36.0%** (45/125) |
| Recall | **4.8%** (4/84) |
| Precision | **100%** (0 false positives) |
| Specificity | **100%** (41/41 human correct) |
| Provider attribution | **4/4 correct** |

The earlier 100% was real but measured only Gemini — the one provider with a
clean C2PA implementation. Widening to Suno and Udio dropped recall by 95 points.

> One false *hint*: an FMA human track surfaced an “OpenAI” vendor string from
> random audio bytes. It stayed a hint and never became a positive, so precision
> held — but it is the false-positive class flagged in §5.

## 3. Why recall collapsed — and what it means

`run_provenance_survival.py` re-encodes files that **are** known to carry a
signed manifest, then re-reads them:

| Operation | Manifest survives | Models |
| --- | --- | --- |
| original | **4/4 (100%)** | straight from the generator |
| MP3 320 kbps | **0/4 (0%)** | any upload pipeline |
| MP3 192 kbps | **0/4 (0%)** | streaming delivery |
| MP3 128 kbps | **0/4 (0%)** | lossy redistribution |
| PCM WAV bounce | **0/4 (0%)** | a DAW export |
| ID3 tag removed | **0/4 (0%)** | a one-line strip |

**Every** operation destroys it — including 320 kbps, which is sonically
transparent, and a lossless WAV bounce. C2PA lives in the container, and
re-encoding rebuilds the container.

This also explains the Suno/Udio result. Those tracks were re-encoded with
LAME 3.100 by the dataset authors and carry no ID3 tag at all, so **their 0%
detection shows what survives collection and redistribution, not that Suno and
Udio never disclosed.** That distinction matters and should not be dropped when
this number is quoted.

The strategic consequence is the strongest argument in the deck:

- Metadata-based provenance works only on first-party downloads.
- A watermark lives in the samples, so none of the six operations above touch it.
- Google already does both: a C2PA manifest **and** an inaudible SynthID
  watermark. The manifest dies on transcode; the watermark survives — and we
  cannot read it, because Google does not release the detector.

## 4. Watermark robustness — the decoder fails on MP3

3 carriers × 26 perturbations = 78 trials, scoring exact record-ID recovery.

| Decoder | Survived | Rate |
| --- | --- | --- |
| `baseline` (shipping) | 53/78 | **68.0%** |
| `sync_search` (proposed) | 75/78 | **96.2%** |

**Every MP3 bitrate defeats the shipping decoder, including 320 kbps.**

| Perturbation | Baseline | Sync search | Base z | Sync z |
| --- | --- | --- | --- | --- |
| mp3_320 / 192 / 128 / 96 | **FAIL** | PASS | 1.19 | 11.6–13.0 |
| compression + normalize + MP3 | **FAIL** | PASS | 1.17 | 14.59 |
| crop first 0.5 s | **FAIL** | PASS | 1.06 | 14.12 |
| insert 1000-sample lead-in | **FAIL** | PASS | 1.00 | 14.35 |
| 8 kHz low-pass | 67% | PASS | 3.04 | 6.75 |
| 1% tempo change | **FAIL** | **FAIL** | 1.03 | 1.24 |
| *(17 others: noise to 20 dB SNR, ±6 dB gain, 8-bit requantization, resampling, compression, clipping, echo, layering, fades)* | PASS | PASS | | |

The watermark is not the problem. MP3 adds ~1800 samples of codec delay; the
payload survives intact (z ≈ 13, exact ID recovered) but the shipping decoder
assumes it starts at sample 0.

**The fix is contained.** A shift of *d* samples is `q·BIT_LEN + r`, so sweeping
sub-bit offset *r* over `[0, BIT_LEN)` and payload phase `q mod NBITS` covers
every misalignment. One FFT cross-correlation gets all offsets at once. Runtime
**0.07–0.28 s** per 30 s. Implemented in `eval/detectors.py`, deliberately **not**
merged — changing the shipping decoder is the team's call.

Genuinely fatal: a **1% tempo change**, which rescales the bit grid.

## 5. Benchmark against Meta AudioSeal

AudioSeal is the one useful comparison: a deployed audio watermark with a
**public** detector, unlike SynthID.

| | Audiomark | AudioSeal |
| --- | --- | --- |
| Embedding SNR | **20.1 dB** | **36.6 dB** |
| Detects its own mark | yes | yes (p = 1.00) |
| Detects the other's | **no** | **no** (p = 0.13) |
| Clean-audio false positive | no | no (p = 0.06) |

**We are blind to AudioSeal.** It ships in Meta's audio models and is already in
our provider catalog, so AudioSeal-watermarked audio is a live coverage gap. Its
detector is open, so this is fixable — at the cost of a torch dependency, which
is a real question for a Vercel deployment.

**Our watermark is ~16 dB louder than Meta's.** That buys robustness — we beat
AudioSeal on 20 dB SNR noise (it drops to p = 0.51) — but 20 dB under the music
is loud enough that nobody should call it imperceptible without a listening test.

On matched perturbations at 16 kHz, sync search and AudioSeal both survive MP3,
cropping, shifting, low-pass and compression; both fail the 1% tempo change.

## 6. Smaller findings

**Vendor hints fire on random audio bytes.** The scan reads the whole file,
audio payload included. Two false hits across the corpus (“Meta” on a Gemini
track, “OpenAI” on an FMA human track). Thirteen of 47 aliases are ≤5 characters.
Scoping the scan to metadata regions removes the class.

**32-bit float WAVs bypass the primary reader.** Stdlib `wave` rejects them;
detection falls back to miniaudio and still works. The fallback is load-bearing
for a format DAWs export by default.

**A bug in our own harness, for the record.** The first AudioSeal run showed MP3
destroying both watermarks. That was wrong: 192 kbps is illegal at 16 kHz, so
LAME silently switched layers and resampled to 44.1 kHz — a 2.8× time-scale
change reported as codec damage. `legal_bitrate()` now clamps to the rate's MPEG
ladder and the decoded rate is converted back. The 44.1 kHz results were never
affected.

## 7. What is still unmeasured

| Source | Status |
| --- | --- |
| Gemini, Suno, Udio, human | **done** — 125 files |
| ElevenLabs | needs a subscription |
| OpenAI audio + Content Provenance API | needs an API key |
| SynthID decoding | **not possible** — detector unreleased; needs the Google partnership |
| Claude | not applicable — Claude does not generate audio |
| Suno/Udio *at source* | needs first-party downloads, not a redistributed dataset |

That last row is the highest-value remaining test. Everything we know about
Suno and Udio comes from tracks already stripped by redistribution; whether they
disclose at generation time is still an open question, and it is the one that
decides whether reading manifests is worth anything for those providers.

## 8. Recommended next steps

1. **Lead with the survival result.** “Provenance metadata does not survive a
   single transcode” is the clearest argument for watermarking, and it is now
   measured rather than asserted.
2. **Ship `c2patool` + trust list.** Without it *verified* is unreachable.
3. **Decide on sync search.** 68% → 96%, written and tested.
4. **Get first-party Suno/Udio downloads.** One paid month settles the open
   question in §7.
5. **Scope the vendor-hint scan to metadata regions.**
6. **Listening test at 20 dB SNR** — we are 16 dB louder than Meta.
7. **Consider an AudioSeal detector**, weighing the torch dependency.

## 9. What changed

| File | Change |
| --- | --- |
| `cli/c2pa_jumbf.py` | **New** — JUMBF/CBOR C2PA reader |
| `cli/audiomark.py` | Scanner reads real C2PA; verdict reflects declared AI |
| `cli/provenance_core.py` | Manifest signal carries `declaresAiGenerated` + evidence |
| `cli/test_c2pa_jumbf.py` | **New** — 43 tests |
| `eval/` | **New** — 4 harnesses, corpus fetcher, 26 perturbations, detectors, metrics, labels |

Tests: **55 passed** (43 new, 12 pre-existing, no regressions).

Corpus (~87 MB, not committed) is rebuilt from public sources by
`eval/fetch_corpus.py`, which pulls samples out of multi-gigabyte remote ZIPs
over HTTP range requests rather than downloading ~45 GB of archives.
