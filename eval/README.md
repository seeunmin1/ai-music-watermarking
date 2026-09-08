# Evaluation harness

Two harnesses, measuring two different things. Keeping them apart matters,
because the product has two modes and they fail for unrelated reasons.

| Harness | Mode | Question |
| --- | --- | --- |
| `run_detection_eval.py` | A — Detect | Given someone else's file, do we reach the right statutory verdict? |
| `run_provenance_survival.py` | A — Detect | Does a manifest survive the trip to a listener? |
| `run_robustness.py` | B — Embed | Does *our* watermark survive the release chain? |
| `audioseal_probe.py` | B — Embed | How do we compare to a deployed third-party watermark? |

`fetch_corpus.py` builds the labeled corpus from public datasets first.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pytest lameenc      # lameenc powers the MP3 round-trips
```

Trusted C2PA validation needs `c2patool` (from `contentauth/c2pa-rs`). Without
it, detection still reads manifests natively but cannot promote a provider to
**verified** — the ceiling is "declared, signature not validated".

```bash
export AUDIOMARK_C2PATOOL_PATH=/path/to/c2patool
export AUDIOMARK_C2PA_TRUST_ANCHORS=/path/to/C2PA-TRUST-LIST.pem
```

## Building the corpus

```bash
.venv/bin/pip install certifi lameenc          # range requests + MP3 round-trips
.venv/bin/python eval/fetch_corpus.py --suno 40 --udio 40 --human 40
```

Pulls Suno and Udio tracks from **SONICS** (ICLR 2025) and human controls from
**FMA small**, into `~/ai-music-corpus/`. Both ship as multi-gigabyte ZIPs, so
members are fetched individually over HTTP range requests (`remote_zip.py`)
rather than downloading ~45 GB. Sampling is seeded and reproducible.

> **Interpret these files carefully.** SONICS tracks were re-encoded by the
> dataset authors (LAME 3.100, no ID3 tag). Absent provenance therefore shows
> what survives redistribution — *not* that the provider failed to disclose at
> generation time. For that question you need first-party downloads.

## Detection accuracy

```bash
.venv/bin/python eval/run_detection_eval.py
.venv/bin/python eval/run_detection_eval.py --json eval/results/detection.json
.venv/bin/python eval/run_detection_eval.py --root downloads=/path/to/audio
```

The report separates two things that get conflated:

- **Detection accuracy** — of the files carrying a provenance payload, how many
  did we read correctly? A property of our code.
- **Corpus coverage** — how many AI files carry a payload at all? A property of
  the ecosystem. No provenance-based tool can detect a file that carries
  nothing, so this number bounds recall no matter how good the code is.

Accuracy is reported under three **verdict policies**, because "is this AI?" has
no single technical answer — it depends on the claim level you act on:

| Policy | Counts as an AI call | Use |
| --- | --- | --- |
| `strict` | `ai_generated` only | Claims that must hold up in a dispute |
| `disclosure` | + `c2pa_detected_untrusted` | Platform enforcing a disclosure rule |
| `permissive` | + `probably_ai_generated` | Triage / review queues |

## Watermark robustness

```bash
.venv/bin/python eval/run_robustness.py --duration 30
.venv/bin/python eval/run_robustness.py --json eval/results/robustness.json
.venv/bin/python eval/run_robustness.py --only mp3_192 crop_0s5
```

Embeds a known record ID into each carrier, applies 26 perturbations covering
codecs, gain, EQ, dynamics, noise, resampling, timeline edits and layering, then
tries to decode the payload back.

Two decoders run on identical audio, which is what makes the result actionable:

- `baseline` — the shipping decoder, which assumes the payload starts at sample 0.
- `sync_search` — a proposed decoder that locates the payload before decoding.

Where `baseline` fails and `sync_search` succeeds, **the watermark survived and
the decoder lost alignment** — a decoder fix, not a watermark redesign. Where
both fail, the signal itself did not survive.

`sync_search` lives here rather than in `cli/` on purpose: it is a validated
proposal, not a shipped change. See `../REPORT.md` for what it would buy.

## Provenance survival

```bash
.venv/bin/python eval/run_provenance_survival.py
```

Takes files that **do** carry a signed manifest, re-encodes them the way an
upload pipeline would, and re-reads them. Isolates "did the provider disclose?"
from "is the disclosure still here?" — the two get conflated constantly, and
only the second is a property of the distribution chain.

## AudioSeal benchmark

```bash
.venv/bin/pip install audioseal
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
.venv/bin/python eval/audioseal_probe.py --duration 40
```

Meta's AudioSeal is the one third-party scheme measurable end to end, since its
detector is public. Reports cross-detection (can either tool read the other's
mark?), embedding SNR, and matched robustness at 16 kHz.

The `SSL_CERT_FILE` export is needed because framework Python on macOS ships no
CA bundle and `torch.hub` downloads the weights.

## Growing the validation set

`labels.json` is declarative, so scaling to the 100–1000 file target needs no
code change. Drop files into a directory and point a `collections` entry at it:

```json
{
  "id_prefix": "suno",
  "root": "corpus",
  "glob": "suno/*.*",
  "label": "ai",
  "cohort": "third_party_ai",
  "provider": "Suno",
  "expected_provenance": "unknown"
}
```

Roots resolve from `labels.json`, `--root name=path`, or
`AUDIOMARK_EVAL_ROOT_<NAME>`. Entries whose files are absent are reported as
missing rather than silently skipped.

**One trap worth stating explicitly.** The `human` cohort must be genre-matched
to the AI cohort. A set of AI pop tracks against a set of classical piano
recordings is separable by instrumentation and recording conditions alone, so
any classifier trained on it will score well while having learned nothing about
provenance. That number would not survive contact with a real catalogue.

## Labels

Three ground-truth classes:

- `ai` — fully generated by a model.
- `human` — no generative AI in the signal chain.
- `ai_assisted` — human-produced with some AI contribution. Scored **separately**
  and excluded from the binary metrics: a track with an AI-rendered stem is a
  different statutory question from a fully generated one, and folding it into
  either class misstates the result.

## Tests

```bash
.venv/bin/python -m pytest cli/test_c2pa_jumbf.py cli/test_provenance_core.py -q
```
