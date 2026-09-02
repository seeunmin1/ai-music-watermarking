import { useCallback, useEffect, useRef, useState } from "react";
import {
  embedWatermark, detectWatermark, scanMetadata, readFileAll, encodeWav, sha256Hex,
} from "./dsp.js";
import { loadState, saveState, buildManifest } from "./persistence.js";
import { analyzeProvenance, VENDOR_LABELS } from "./provenance.js";

const DETECTION_API_URL = import.meta.env.VITE_PROVENANCE_API_URL || "/api";
const EMPTY_WATERMARK = { found: false, recordId: 0, z: 0, confidence: 0, repetitions: 0 };

function normalizeBackendDetection(detection) {
  if (!detection) return null;
  return {
    found: !!detection.found,
    recordId: Number(detection.recordId ?? detection.record_id ?? 0),
    z: Number(detection.z ?? 0),
    confidence: Number(detection.confidence ?? 0),
    repetitions: Number(detection.repetitions ?? 0),
  };
}

function normalizeBackendMetadata(metadata) {
  if (!metadata) return null;
  return {
    id3: !!metadata.id3,
    jumbf: !!metadata.jumbf,
    manifest: metadata.manifest || null,
    fields: metadata.fields || null,
    vendorHints: metadata.vendorHints || metadata.vendor_hints || [],
  };
}

function mergeMetadata(localMeta, backendMeta) {
  if (!backendMeta) return localMeta;
  return {
    id3: localMeta.id3 || backendMeta.id3,
    jumbf: localMeta.jumbf || backendMeta.jumbf,
    manifest: localMeta.manifest || backendMeta.manifest,
    fields: localMeta.fields || backendMeta.fields,
    vendorHints: Array.from(new Set([...(localMeta.vendorHints || []), ...(backendMeta.vendorHints || [])])),
  };
}

/* ================================================================
   AUDIOMARK AI — Compliance Engine MVP
   CA SB 942 / AB 853 — dual-mode detect/embed portal.

   Visual design adapted from a companion prototype; the watermark
   engine underneath (spread-spectrum DSSS mark + C2PA-style in-file
   manifest, byte-compatible with cli/audiomark.py) is real and runs
   entirely client-side — see src/dsp.js.
   ================================================================ */

function DropZone({ label, sublabel, fileName, onFile, disabled }) {
  const [over, setOver] = useState(false);
  const inputRef = useRef(null);
  return (
    <div
      className={"drop-zone" + (over ? " over" : "")}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault(); setOver(false);
        if (!disabled && e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
      }}
    >
      <span className="drop-icon">+</span>
      <span className="drop-title">{fileName || label}</span>
      <span className="drop-subtitle">{sublabel}</span>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.m4a,.flac"
        hidden
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
    </div>
  );
}

function Waveform({ samples }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !samples) return;
    const w = (cv.width = cv.offsetWidth * 2), h = (cv.height = 96);
    const g = cv.getContext("2d");
    g.clearRect(0, 0, w, h);
    g.strokeStyle = "#0d9488";
    g.lineWidth = 1.4;
    const step = Math.max(1, Math.floor(samples.length / (w / 2)));
    g.beginPath();
    for (let x = 0; x < w / 2; x++) {
      let mn = 1, mx = -1;
      for (let i = x * step; i < Math.min((x + 1) * step, samples.length); i++) {
        if (samples[i] < mn) mn = samples[i];
        if (samples[i] > mx) mx = samples[i];
      }
      g.moveTo(x * 2, h / 2 + mn * h * 0.46);
      g.lineTo(x * 2, h / 2 + mx * h * 0.46 + 0.8);
    }
    g.stroke();
  }, [samples]);
  return <canvas ref={ref} style={{ width: "100%", height: 96, display: "block", borderRadius: 8, background: "#f8fafc", marginTop: 10 }} />;
}

export default function AudiomarkAI() {
  const [mode, setMode] = useState("detect"); // detect | embed
  const [registry, setRegistry] = useState([]);
  const [audit, setAudit] = useState([]);
  const [manifestEnabled, setManifestEnabled] = useState(true);
  const [form, setForm] = useState({ provider: "Audiomark Labs", system: "DemoTTS", version: "1.0" });
  const [backendStatus, setBackendStatus] = useState(DETECTION_API_URL ? "checking" : "browser_only");

  useEffect(() => {
    setRegistry(loadState("registry", []));
    setAudit(loadState("audit", []));
  }, []);

  useEffect(() => {
    if (!DETECTION_API_URL) return;
    let cancelled = false;
    fetch(`${DETECTION_API_URL.replace(/\/$/, "")}/health`)
      .then((res) => {
        if (!cancelled) setBackendStatus(res.ok ? "connected" : "unreachable");
      })
      .catch(() => {
        if (!cancelled) setBackendStatus("unreachable");
      });
    return () => { cancelled = true; };
  }, []);

  const log = useCallback((type, detail) => {
    setAudit((prev) => {
      const next = [{ t: new Date().toISOString(), type, detail }, ...prev].slice(0, 200);
      saveState("audit", next);
      return next;
    });
  }, []);

  const fmtT = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

  /* ---- DETECT ---- */
  const [dFile, setDFile] = useState(null);
  const [dBusy, setDBusy] = useState(false);
  const [dRes, setDRes] = useState(null);

  const onDetectFile = async (f) => {
    setDRes(null); setDBusy(true);
    try {
      const d = await readFileAll(f);
      setDFile({ file: f, name: f.name, ...d, browserDecodeError: null });
    } catch (error) {
      try {
        const buf = await f.arrayBuffer();
        setDFile({
          file: f,
          name: f.name,
          bytes: new Uint8Array(buf),
          samples: null,
          sampleRate: null,
          duration: null,
          hash: await sha256Hex(buf.slice(0)),
          browserDecodeError: error?.message || "Browser audio preview decode failed.",
        });
      } catch {
        setDRes({ error: "Could not read that audio file." });
      }
    }
    setDBusy(false);
  };

  const runDetect = async () => {
    if (!dFile) return;
    setDBusy(true); setDRes(null);
    await new Promise((r) => setTimeout(r, 250)); // let the "Checking..." state read on screen
    try {
      let wm = dFile.samples ? detectWatermark(dFile.samples) : EMPTY_WATERMARK;
      let meta = scanMetadata(dFile.bytes);
      let provenance = analyzeProvenance({ wm, meta, registry });
      let audioDecode = dFile.samples
        ? { status: "decoded", codec: "browser", decoder: "Web Audio API", sample_rate: dFile.sampleRate, duration: dFile.duration, detail: "Browser decoded audio for local preview checks." }
        : { status: "unsupported", codec: "browser", decoder: "Web Audio API", sample_rate: null, duration: null, detail: dFile.browserDecodeError || "Browser audio decode failed." };
      let classifier = { status: "not_configured", candidates: [], detail: "Backend local attribution was not run." };
      let backendUsed = false;
      if (DETECTION_API_URL) {
        try {
          const fd = new FormData();
          fd.append("file", dFile.file);
          const backend = await fetch(`${DETECTION_API_URL.replace(/\/$/, "")}/detect`, { method: "POST", body: fd });
          if (backend.ok) {
            const enriched = await backend.json();
            const backendWm = normalizeBackendDetection(enriched?.detection);
            const backendMeta = normalizeBackendMetadata(enriched?.metadata);
            if (backendWm) wm = backendWm;
            meta = mergeMetadata(meta, backendMeta);
            if (enriched?.provenance) provenance = enriched.provenance;
            if (enriched?.audio_decode) audioDecode = enriched.audio_decode;
            if (enriched?.classifier) classifier = enriched.classifier;
            backendUsed = true;
            setBackendStatus("connected");
          }
        } catch {
          provenance.limitations = [...provenance.limitations, "Configured backend detection API was unreachable; browser-only checks were used."];
          setBackendStatus("unreachable");
        }
      }
      const aiDetected = provenance.generationClaimLevel === "verified";
      setDRes({ wm, meta, provenance, aiDetected, audioDecode, classifier, backendUsed });
      log("verify", provenance.generationClaimLevel === "verified"
        ? `AI-GENERATED verdict for "${dFile.name}" resolved via ${provenance.resolvedVia}`
        : `${provenance.displayTitle.toUpperCase()} for "${dFile.name}" (z=${wm.z.toFixed(1)}, ${provenance.displaySubtitle})`);
    } catch { setDRes({ error: "Detection failed on this file." }); }
    setDBusy(false);
  };

  /* ---- EMBED ---- */
  const [src, setSrc] = useState(null);
  const [eBusy, setEBusy] = useState(false);
  const [eRes, setERes] = useState(null);
  const [eErr, setEErr] = useState("");

  const onEmbedFile = async (f) => {
    setEErr(""); setERes(null); setEBusy(true);
    try { const d = await readFileAll(f); setSrc({ name: f.name, ...d }); }
    catch { setEErr("Could not decode that file as audio."); }
    setEBusy(false);
  };

  const doEmbed = async () => {
    if (!src) return;
    setEBusy(true); setEErr("");
    await new Promise((r) => setTimeout(r, 250));
    try {
      const recordId = Math.floor(Math.random() * 0xffffff);
      const em = embedWatermark(src.samples, recordId);
      if (!em) { setEErr("Audio must be at least ~2.1 s to carry one payload frame."); setEBusy(false); return; }
      const rec = {
        idHex: recordId.toString(16).padStart(6, "0"),
        provider: form.provider, system: form.system, version: form.version,
        timestamp: new Date().toISOString(), uid: crypto.randomUUID(),
        fileName: src.name, sourceHash: src.hash, repetitions: em.repetitions,
        status: "active", manifestEmbedded: manifestEnabled,
      };
      const extraChunks = [];
      let manUrl = null;
      if (manifestEnabled) {
        const manifest = buildManifest(rec);
        const manBytes = new TextEncoder().encode(JSON.stringify(manifest));
        extraChunks.push({ id: "c2pa", data: manBytes });
        manUrl = URL.createObjectURL(new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" }));
      }
      const wavBlob = encodeWav(em.samples, src.sampleRate, extraChunks);
      rec.wavHash = await sha256Hex(await wavBlob.arrayBuffer());
      const wavUrl = URL.createObjectURL(wavBlob);
      setRegistry((prev) => { const next = [rec, ...prev]; saveState("registry", next); return next; });
      log("embed", `Embedded record ${rec.idHex}${manifestEnabled ? " + in-file C2PA manifest" : " (manifest disclosure off — latent mark only)"} into "${src.name}" (${em.repetitions}x redundancy) for ${form.provider}/${form.system} v${form.version}`);
      setERes({ rec, wavUrl, manUrl, wmSamples: em.samples });
    } catch (e) { setEErr("Embedding failed: " + e.message); }
    setEBusy(false);
  };

  const resetDetect = () => { setDFile(null); setDRes(null); };
  const resetEmbed = () => { setSrc(null); setERes(null); setEErr(""); };

  /* ---- REVOCATION ---- */
  const revoke = (rec) => {
    const deadline = new Date(Date.now() + 96 * 3600 * 1000).toISOString();
    setRegistry((prev) => {
      const next = prev.map((r) => (r.uid === rec.uid ? { ...r, status: "revocation_pending", revokedAt: new Date().toISOString(), revocationDeadline: deadline } : r));
      saveState("registry", next);
      return next;
    });
    log("revocation", `Strip/violation flagged on record ${rec.idHex} ("${rec.fileName}"). 96-hour revocation window opened — deadline ${deadline}`);
  };

  /* ---- FEEDBACK (real: appends to the local audit log) ---- */
  const [feedbackText, setFeedbackText] = useState("");
  const [feedbackOptIn, setFeedbackOptIn] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);

  const sendFeedback = (e) => {
    e.preventDefault();
    if (!feedbackText.trim()) return;
    log("feedback", `${feedbackText.trim()}${feedbackOptIn ? " [opted in to follow-up]" : ""}`);
    setFeedbackSent(true);
    setFeedbackText("");
    setTimeout(() => setFeedbackSent(false), 2500);
  };

  const complianceItems = [
    ["Free, local tool", "Detect and embed run entirely in your browser — no account, no paid gate"],
    ["Byte-compatible CLI", "cli/audiomark.py mirrors this scheme exactly — files embedded by one decode in the other"],
    ["Latent disclosure", "Provider, system version, timestamp, and unique ID resolved via the registry"],
    ["Manifest option", "The toggle to the left controls whether the in-file C2PA manifest is written"],
    ["No retention", "Audio is processed in memory and never leaves the browser"],
    ["Feedback loop", "Opt-in feedback below is logged to the local audit trail"],
  ];

  const backendLabel = {
    checking: "Backend checking",
    connected: "Backend model connected",
    unreachable: "Backend unavailable",
    browser_only: "Browser-only mode",
  }[backendStatus];

  const liveResult = dRes && !dRes.error ? {
    verdict: dRes.provenance.verdict,
    claim_level: dRes.provenance.claimLevel,
    generation_claim_level: dRes.provenance.generationClaimLevel,
    provider_claim_level: dRes.provenance.providerClaimLevel,
    provider_verification_status: dRes.provenance.providerVerificationStatus,
    provider: dRes.provenance.provider,
    attributed_provider: dRes.provenance.attributedProvider,
    attributed_system: dRes.provenance.attributedSystem,
    provider_candidates: dRes.provenance.providerCandidates,
    system: dRes.provenance.system,
    content_id: dRes.provenance.contentId,
    created_at: dRes.provenance.createdAt,
    confidence: dRes.provenance.confidence,
    attribution_confidence: dRes.provenance.attributionConfidence,
    resolved_via: dRes.provenance.resolvedVia,
    attribution_resolved_via: dRes.provenance.attributionResolvedVia,
    official_check_status: dRes.provenance.officialCheckStatus,
    backend_used: dRes.backendUsed,
    audio_decode: dRes.audioDecode,
    classifier: dRes.classifier,
    watermark: { found: dRes.wm.found, z: Number(dRes.wm.z.toFixed(2)), confidence: Number(dRes.wm.confidence.toFixed(1)), repetitions: dRes.wm.repetitions },
    signals: dRes.provenance.signals,
    limitations: dRes.provenance.limitations,
  } : null;

  const provenanceFields = dRes && !dRes.error ? [
    ["Verdict", dRes.provenance.displayTitle],
    ["Generation claim", dRes.provenance.generationClaimLevel],
    ["Provider claim", dRes.provenance.providerClaimLevel],
    ["Provider verification", dRes.provenance.providerVerificationStatus],
    ["Provider / Platform", dRes.provenance.attributedProvider || dRes.provenance.provider || "Unknown"],
    ["System version", dRes.provenance.attributedSystem || dRes.provenance.system || "Unknown"],
    ["Audio decode", dRes.audioDecode?.status || "not run"],
    ["Local classifier", dRes.classifier?.status || "not run"],
    ["Timestamp", fmtT(dRes.provenance.createdAt)],
    ["Unique content ID", dRes.provenance.contentId || "—"],
    ["Resolved via", dRes.provenance.attributionResolvedVia || dRes.provenance.resolvedVia],
    ["Record status", dRes.provenance.record ? (dRes.provenance.record.status === "active" ? "Active" : "⚠ In 96h revocation window") : "External or unregistered payload"],
  ] : null;

  return (
    <main>
      <section className="hero-shell">
        <nav className="topbar" aria-label="Primary">
          <a className="brand" href="#verify" aria-label="Audiomark AI home">
            <span className="brand-mark">A</span>
            <span>Audiomark AI</span>
          </a>
          <div className="nav-actions">
            <a href="#signals">Signals</a>
            <a href="#registry">Registry</a>
            <a className="nav-button" href="#verify">Verify</a>
          </div>
        </nav>

        <div className="hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">Third-party AI audio screening</p>
            <h1>Evidence-based detection for probable AI-generated audio.</h1>
            <p className="hero-text">
              Upload audio to combine backend model analysis with provenance checks for
              watermark payloads, C2PA manifests, provider hints, and local attribution.
              Results stay explicit about verified, detected, probable, hint-only, and unknown evidence.
            </p>
            <div className="hero-metrics" aria-label="Compliance highlights">
              <div><strong>5</strong><span>evidence levels</span></div>
              <div><strong>MP3</strong><span>backend decode path</span></div>
              <div><strong>0s</strong><span>content retention</span></div>
            </div>
          </div>

          <section className="verify-panel" id="verify" aria-labelledby="verify-title">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">{mode === "detect" ? "Detection Portal" : "Embed Portal"}</p>
                <h2 id="verify-title">{mode === "detect" ? "Verify audio provenance" : "Create a disclosure record"}</h2>
              </div>
              <span className={`status-pill backend-${backendStatus}`}>{mode === "detect" ? backendLabel : "No content retained"}</span>
            </div>

            <div className="segmented" role="tablist" aria-label="Mode">
              <button className={mode === "detect" ? "active" : ""} type="button" onClick={() => setMode("detect")} aria-selected={mode === "detect"}>Detect</button>
              <button className={mode === "embed" ? "active" : ""} type="button" onClick={() => setMode("embed")} aria-selected={mode === "embed"}>Embed</button>
            </div>

            {mode === "detect" && (
              <>
                <div className="verify-form">
                  <DropZone
                    label="Select an audio file"
                    sublabel="WAV, MP3, M4A, FLAC — backend model when connected, browser fallback otherwise"
                    fileName={dFile?.name}
                    onFile={onDetectFile}
                    disabled={dBusy}
                  />
                  {dFile?.samples && <Waveform samples={dFile.samples} />}
                  {dFile?.browserDecodeError && (
                    <p className="inline-warning">Browser preview decode failed; the backend can still analyze this upload if connected.</p>
                  )}
                  <div style={{ display: "flex", gap: 10 }}>
                    <button className="primary-button" type="button" onClick={runDetect} disabled={!dFile || dBusy} style={{ flex: 1 }}>
                      {dBusy ? "Checking…" : "Run AI audio check"}
                    </button>
                    {(dFile || dRes) && <button className="ghost-button" type="button" onClick={resetDetect}>Clear</button>}
                  </div>
                </div>

                {dRes?.error && <div className="result-box bad"><div><span className="result-label">Error</span><p>{dRes.error}</p></div></div>}

                {dRes && !dRes.error && (
                  <div className={"result-box visible" + (!dRes.aiDetected ? "" : "")}>
                    <div>
                      <span className="result-label">{dRes.provenance.displayTitle}</span>
                      <strong>{dRes.provenance.displaySubtitle}</strong>
                      <p>{dFile?.name} — {dRes.audioDecode?.status || "decode not run"} · {dRes.classifier?.status || "classifier not run"} · {dRes.provenance.resolvedVia || dRes.provenance.attributionResolvedVia || "not resolved"}</p>
                    </div>
                    <div className="confidence-ring" style={{ "--pct": String(dRes.provenance.confidence) }}>
                      {dRes.provenance.attributionConfidence || dRes.provenance.confidence}
                    </div>
                  </div>
                )}
              </>
            )}

            {mode === "embed" && (
              <>
                <div className="verify-form">
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                    {[["provider", "Provider"], ["system", "System"], ["version", "Version"]].map(([k, l]) => (
                      <label className="field" key={k}>
                        {l}
                        <input value={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.value })} />
                      </label>
                    ))}
                  </div>
                  <DropZone
                    label="Drop generated audio here, or click to browse"
                    sublabel="WAV, MP3, M4A, FLAC — processed in memory, zero permanent storage"
                    fileName={src?.name}
                    onFile={onEmbedFile}
                    disabled={eBusy}
                  />
                  {src && <Waveform samples={eRes ? eRes.wmSamples : src.samples} />}
                  <div style={{ display: "flex", gap: 10 }}>
                    <button className="primary-button" type="button" onClick={doEmbed} disabled={!src || eBusy || !!eRes} style={{ flex: 1 }}>
                      {eBusy ? "Embedding…" : "Embed watermark"}
                    </button>
                    {(src || eRes) && <button className="ghost-button" type="button" onClick={resetEmbed}>{eRes ? "New file" : "Clear"}</button>}
                  </div>
                </div>

                {eErr && <div className="result-box bad"><div><span className="result-label">Error</span><p>{eErr}</p></div></div>}

                {eRes && (
                  <div className="result-box visible">
                    <div>
                      <span className="result-label">Watermark embedded</span>
                      <strong>Record {eRes.rec.idHex}</strong>
                      <p>{eRes.rec.repetitions}× redundancy · <a className="dl-link" href={eRes.wavUrl} download={src.name.replace(/\.[^.]+$/, "") + ".audiomark.wav"}>↓ WAV</a>{eRes.manUrl && <> · <a className="dl-link" href={eRes.manUrl} download="manifest.c2pa.json">↓ manifest</a></>}</p>
                    </div>
                  </div>
                )}
              </>
            )}
          </section>
        </div>
      </section>

      <section className="content-band">
        <div className="section-heading">
          <div>
            <p className="eyebrow">System Provenance</p>
            <h2>Detected statutory payload</h2>
          </div>
        </div>
        {provenanceFields ? (
          <div className="provenance-grid">
            {provenanceFields.map(([label, value]) => (
              <article className="info-card" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </article>
            ))}
          </div>
        ) : (
          <p className="empty-note">Run <b>AI audio check</b> above on a file to see resolved provenance fields here.</p>
        )}
      </section>

      <section className="split-band" id="compliance">
        <div>
          <p className="eyebrow">Provider Console</p>
          <h2>Embed disclosures at generation time</h2>
          <p>
            Every embed creates a signed record, embeds a latent audio marker, and writes a
            manifest disclosure into the file — unless you turn it off below, in which case the
            output carries only the latent mark (still resolvable via the registry).
          </p>

          <div className="toggle-row">
            <div>
              <strong>Manifest disclosure</strong>
              <span>Writes a C2PA-style manifest into the next embedded file</span>
            </div>
            <button
              className={manifestEnabled ? "toggle on" : "toggle"}
              type="button"
              onClick={() => setManifestEnabled(!manifestEnabled)}
              aria-pressed={manifestEnabled}
            >
              <span />
            </button>
          </div>
        </div>

        <div className="check-grid">
          {complianceItems.map(([title, detail]) => (
            <article className="check-card" key={title}>
              <span className="check-dot" />
              <h3>{title}</h3>
              <p>{detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="content-band white-band" id="signals">
        <div className="section-heading">
          <p className="eyebrow">Live Result</p>
          <h2>Signal breakdown</h2>
        </div>

        <div className="api-grid">
          <div className="code-panel">
            <div className="code-tabs">
              <span>last detection result</span>
              <span>JSON</span>
            </div>
            <pre>{liveResult ? JSON.stringify(liveResult, null, 2) : "// Run Detect above — this shows the real, live result object,\n// not a canned example."}</pre>
          </div>

          <div className="test-panel">
            <h3>Per-signal status</h3>
            {[
              ["Backend model API", dRes && !dRes.error ? (dRes.backendUsed ? "connected" : backendLabel.toLowerCase()) : backendLabel.toLowerCase()],
              ["Audio decode", dRes && !dRes.error ? `${dRes.audioDecode?.status || "not run"}${dRes.audioDecode?.sample_rate ? ` · ${dRes.audioDecode.sample_rate} Hz` : ""}` : "—"],
              ["Local attribution classifier", dRes && !dRes.error ? (dRes.classifier?.status || "not run") : "—"],
              ["Audiomark SS/v1 latent mark", dRes && !dRes.error ? (dRes.wm.found ? `decoded · ${dRes.wm.repetitions}× reps` : "not detected") : "—"],
              ["Embedded C2PA manifest", dRes && !dRes.error ? (dRes.meta.manifest ? "parsed" : "not found") : "—"],
              ["JUMBF container markers", dRes && !dRes.error ? (dRes.meta.jumbf ? "present" : "not found") : "—"],
              ["ID3v2 metadata block", dRes && !dRes.error ? (dRes.meta.id3 ? "present" : "not found") : "—"],
              ["Verified provenance", dRes && !dRes.error ? (dRes.provenance.providerVerificationStatus === "verified" ? `${dRes.provenance.attributedProvider || dRes.provenance.provider} via ${dRes.provenance.attributionResolvedVia || dRes.provenance.resolvedVia}` : "not resolved") : "—"],
              ["C2PA detected", dRes && !dRes.error ? (dRes.provenance.generationClaimLevel === "detected" ? dRes.provenance.displaySubtitle : "not detected") : "—"],
              ["Probable attribution", dRes && !dRes.error ? (dRes.provenance.generationClaimLevel === "probable" ? `${dRes.provenance.attributedProvider || dRes.provenance.provider} via ${dRes.provenance.attributionResolvedVia || dRes.provenance.resolvedVia}` : "not used") : "—"],
              ["Official provider checks", dRes && !dRes.error ? dRes.provenance.officialCheckStatus.status : "—"],
              ["Google SynthID-Audio decoder", dRes && !dRes.error ? (dRes.provenance.signals.some((s) => s.id === "google_synthid" && s.status === "verified") ? "verified payload" : "unsupported official decoder") : "—"],
              ["WMAR clustered-token detector", dRes && !dRes.error ? "experimental backend hook" : "—"],
            ].map(([name, status]) => (
              <div className="test-row" key={name}>
                <span>{name}</span>
                <strong className={status.includes("connected") || status.includes("decoded") || status.includes("parsed") || status.includes("present") || status.includes("verified") ? "pass" : status.includes("unavailable") || status.includes("unsupported") || status.includes("experimental") || status.includes("probable") || status.includes("not_configured") || status.includes("not run") ? "review" : ""}>{status}</strong>
              </div>
            ))}
            {dRes?.audioDecode?.detail && <p className="small-note">{dRes.audioDecode.detail}</p>}
            {dRes?.classifier?.detail && <p className="small-note">{dRes.classifier.detail}</p>}
            {dRes?.meta?.vendorHints?.length > 0 && (
              <div className="test-row">
                <span>Vendor strings found</span>
                <strong>{dRes.meta.vendorHints.map((v) => VENDOR_LABELS[v] || v).join(", ")}</strong>
              </div>
            )}
            {dRes?.provenance?.signals?.filter((s) => s.status === "hint").map((s) => (
              <div className="test-row" key={s.id}>
                <span>{s.category}: {s.label}</span>
                <strong className="review">hint only</strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="content-band" id="registry">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Provenance Registry</p>
            <h2>Records created in this browser</h2>
          </div>
        </div>
        <div className="data-card">
          {registry.length === 0 ? (
            <p className="empty-note">No records yet. Use <b>Embed</b> above to create the first registry entry.</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="data-table">
                <thead><tr><th>ID</th><th>Provider / System</th><th>Created</th><th>File</th><th>Status</th><th></th></tr></thead>
                <tbody>
                  {registry.map((r) => (
                    <tr key={r.uid}>
                      <td className="mono">{r.idHex}</td>
                      <td>{r.provider}<div style={{ color: "var(--muted)", fontSize: 12 }}>{r.system} v{r.version}</div></td>
                      <td>{fmtT(r.timestamp)}</td>
                      <td style={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis" }}>{r.fileName}</td>
                      <td>{r.status === "active"
                        ? <span className="badge">Active</span>
                        : <><span className="badge warn">Revocation</span><div style={{ color: "var(--muted)", fontSize: 12, marginTop: 4 }}>deadline {fmtT(r.revocationDeadline)}</div></>}
                      </td>
                      <td>{r.status === "active" && <button className="link-button" onClick={() => revoke(r)}>Flag strip</button>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="small-note">"Flag strip" simulates licensee strip-monitoring: timestamps the evidence in the audit log and opens the 96-hour statutory revocation window (SB 942 licensee duty).</p>
        </div>
      </section>

      <section className="content-band" id="audit">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Audit Log</p>
            <h2>Every verify, embed, revocation, and feedback event</h2>
          </div>
        </div>
        <div className="data-card">
          {audit.length === 0 ? (
            <p className="empty-note">No events yet.</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="data-table">
                <tbody>
                  {audit.map((e, i) => (
                    <tr key={i}>
                      <td className="mono" style={{ whiteSpace: "nowrap", color: "var(--muted)", fontSize: 11.5 }}>{fmtT(e.t)}</td>
                      <td><span className={"badge" + (e.type === "embed" ? "" : e.type === "revocation" ? " warn" : e.type === "feedback" ? " act" : " dim")}>{e.type.toUpperCase()}</span></td>
                      <td>{e.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="small-note">Only event metadata is logged — never audio content, per the zero-retention design.</p>
        </div>
      </section>

      <section className="feedback-band">
        <div>
          <p className="eyebrow">Feedback Loop</p>
          <h2>Improve detection without collecting unnecessary data</h2>
          <p>Submitting here writes a real entry to the audit log above — nothing is sent anywhere.</p>
        </div>
        <form className="feedback-form" onSubmit={sendFeedback}>
          <label>
            Detection feedback
            <textarea
              placeholder="False positive, missed watermark, or file transformation details"
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
            />
          </label>
          <label className="checkbox-label">
            <input type="checkbox" checked={feedbackOptIn} onChange={(e) => setFeedbackOptIn(e.target.checked)} />
            <span>Opt in to follow-up about this report</span>
          </label>
          <button className="secondary-button" type="submit">{feedbackSent ? "Feedback received" : "Send feedback"}</button>
        </form>
      </section>
    </main>
  );
}
