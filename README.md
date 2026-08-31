# Audiomark AI

Compliance-engine MVP for CA SB 942 / AB 853 statutory AI-audio disclosure: a
dual-mode **Detect / Embed** portal built around a keyed spread-spectrum
latent watermark and a C2PA-style manifest bound into the file.

Two implementations of the same scheme, byte-compatible so a file embedded
by one decodes in the other:

- **[`web/`](web)** — a React + Vite single-page app. Runs entirely
  client-side (Web Audio API, `crypto.subtle` for SHA-256, no server): drop
  in a file for an instant statutory verdict, or embed a watermark + manifest
  into a WAV. Registry and audit log persist to the browser's `localStorage`.
- **[`cli/audiomark.py`](cli/audiomark.py)** — a Python CLI for the same
  embed/detect/registry/audit operations, for scripting or server-side use.

## What's real vs. stubbed in this MVP

**Real:**
- Mode B (Embed): keyed spread-spectrum latent watermark (12-bit sync + 24-bit
  record ID + 8-bit checksum, level-adaptive gain, repeated across the track)
  plus a C2PA-style manifest written into the output WAV as a RIFF `c2pa`
  chunk, bound by SHA-256 hashes.
- Mode A (Detect): a layered provenance pipeline over embedded C2PA manifests,
  official-provider adapter results, Google/OpenAI SynthID-style payloads,
  Suno-style Content Credentials, ElevenLabs/Audiomark watermarks, a local
  probable-attribution classifier, an experimental WMAR clustered-token
  adapter, and vendor-name hints. Provider identity is only **verified** when a
  recoverable payload, registry match, or official adapter result is present;
  classifier output is labeled **probable** and plain metadata strings are
  reported as hints.
- Instant statutory verdict (AI-Generated vs. Unmarked) with provider /
  system+version / timestamp / unique-ID resolution, backed by a persistent
  registry and audit log.

**Experimental / stubbed (labeled in the UI):** official Google/OpenAI/
ElevenLabs/Suno verification calls until endpoint credentials are configured,
WMAR clustered-token detection from `vendor/nograd-audio-wm` until its Python
ML dependencies/checkpoints are installed, and real Ed25519 signing (the
manifest signature is simulated).

This is a demo/MVP for exploring the statutory watermarking workflow — the
crypto and third-party decoders are not production-grade.

## Running the web app

```bash
cd web
npm install
npm run dev
```

Then open the printed local URL. `npm run build` produces a static
production bundle in `web/dist/` (deployable to any static host — Vercel,
Netlify, GitHub Pages, etc.).

## Running the CLI

```bash
cd cli
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python audiomark.py embed in.wav out.wav --provider "Audiomark Labs" \
    --system DemoTTS --version 1.0 --manifest out.manifest.json
python audiomark.py detect out.wav
python audiomark.py detect out.wav --json
python audiomark.py wmar-info
python audiomark.py wmar-info --model-family musicgen_encodec
python audiomark.py train-attribution ../datasets/audio-provenance
python detection_server.py --host 127.0.0.1 --port 8787
python audiomark.py registry
python audiomark.py registry --revoke <record_id>
python audiomark.py audit
```

Registry and audit records are stored as JSON files next to the script.

The web app can call the local backend for official-provider and classifier
checks when started with `VITE_PROVENANCE_API_URL=http://127.0.0.1:8787`.
Without that variable, it still runs the lightweight in-browser detector.

## Provider catalog

Third-party provider support is driven by JSON profiles in `providers/`.
Adding a new generative audio provider should start with one profile containing
aliases, products, supported media, C2PA issuers, official verification status,
auth requirements, confidence policy, and parser adapter. The backend loads the
catalog dynamically for official-check status, C2PA issuer matching, and
metadata hints.

Official API keys and endpoints stay server-side. Browser scans are lightweight;
the backend performs official provider checks and probable local attribution.
