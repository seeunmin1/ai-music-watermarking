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
- Mode A (Detect): a layered provenance pipeline over trusted C2PA/Content
  Credentials signatures, official-provider adapter results, provider payloads,
  Audiomark watermarks, a local probable-attribution classifier, an experimental
  WMAR clustered-token adapter, and vendor-name hints. Third-party provider
  identity is only **verified** when a trusted C2PA signature or official
  adapter result is present; unchecked manifests are labeled **detected**,
  classifier output is labeled **probable**, and plain metadata strings are
  reported as hints.
- Instant statutory verdict (AI-Generated vs. Unmarked) with provider /
  system+version / timestamp / unique-ID resolution, backed by a persistent
  registry and audit log.

**External / experimental (labeled in the UI):** trusted C2PA validation uses
the external `c2patool` CLI from `contentauth/c2pa-rs` when installed on the
backend host, official provider API calls require endpoint credentials, WMAR
clustered-token detection from `vendor/nograd-audio-wm` requires its Python
ML dependencies/checkpoints, and this app's own demo manifest signing is not a
production signing service.

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
Without that variable, it uses the same-origin `/api` backend when deployed;
if it cannot be reached, the app still runs the lightweight in-browser detector.

## Deploying to Vercel

Deploy this repository as a single Vercel project from the repository root.
The included `vercel.json` builds the Vite frontend in `web/` and deploys the
Python detection functions at `/api/health` and `/api/detect` alongside it.

```bash
vercel
```

In Vercel project settings, leave **Root Directory** at the repository root.
The checked-in `vercel.json` pins the Vite framework, installs frontend
dependencies from `web/`, and supplies the build/output configuration.
`/api/detect` accepts a `multipart/form-data`
upload under the `file` field, uses a temporary file only for the request, and
does not persist serverless audit records. The current function limits uploads
to 4 MB, within Vercel request limits.

Set optional server-side environment variables in Vercel when the corresponding
integrations are available: `OPENAI_API_KEY`, `AUDIOMARK_C2PATOOL_PATH`,
`AUDIOMARK_C2PA_TRUST_ANCHORS`, `AUDIOMARK_C2PA_ALLOWED_LIST`,
`AUDIOMARK_C2PA_TRUST_CONFIG`, plus any provider endpoint or credential
variables used by your configured provider adapters. Do not set
`VITE_PROVENANCE_API_URL` for the normal Vercel deployment: the frontend uses
the deployed same-origin API by default.

### External C2PA verification

Install `c2patool` on the backend host and make it available on `PATH`, or set:

```bash
export AUDIOMARK_C2PATOOL_PATH=/path/to/c2patool
```

The backend runs `c2patool <asset>` to read manifests, `c2patool <asset>
--certs` to inspect signing certificates, and `c2patool <asset> trust` to
validate signatures. By default it uses the official C2PA trust-list URL from
the c2pa-rs docs. Override trust inputs with:

```bash
export AUDIOMARK_C2PA_TRUST_ANCHORS=/path/or/url/to/anchors.pem
export AUDIOMARK_C2PA_ALLOWED_LIST=/path/or/url/to/allowed-list.pem
export AUDIOMARK_C2PA_TRUST_CONFIG=/path/or/url/to/store.cfg
```

If `c2patool` is missing or trust validation fails, matching provider metadata
is reported as detected/untrusted or hint-only, not verified.

## Provider catalog

Third-party provider support is driven by JSON profiles in `providers/`.
Adding a new generative audio provider should start with one profile containing
aliases, products, supported media, C2PA issuers, official verification status,
auth requirements, confidence policy, and parser adapter. The backend loads the
catalog dynamically for official-check status, trusted C2PA issuer matching,
and metadata hints.

Official API keys and endpoints stay server-side. Browser scans are lightweight;
the backend performs official provider checks and probable local attribution.
