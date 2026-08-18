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
- Mode A (Detect): matched-filter watermark detector (z-score) and a
  byte-level metadata scanner that finds/parses embedded C2PA manifests,
  ID3v2 tags, JUMBF markers, and vendor-name hints (Suno / ElevenLabs /
  OpenAI / SynthID strings in metadata).
- Instant statutory verdict (AI-Generated vs. Unmarked) with provider /
  system+version / timestamp / unique-ID resolution, backed by a persistent
  registry and audit log.

**Stubbed (labeled in the UI):** Google SynthID and Meta AudioSeal signal
decoders, real Ed25519 signing (the manifest signature is simulated).

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
python audiomark.py registry
python audiomark.py registry --revoke <record_id>
python audiomark.py audit
```

Registry and audit records are stored as JSON files next to the script.
