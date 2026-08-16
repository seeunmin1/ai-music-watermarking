"use client";

import { FormEvent, useMemo, useState } from "react";

type VerificationMode = "file" | "url";

const provenanceFields = [
  ["Provider", "Audiomark AI"],
  ["GenAI system", "VoxForge Studio"],
  ["System version", "2.4.1"],
  ["Created", "Aug 16, 2026, 3:42 PM PDT"],
  ["Unique ID", "amk_7Q9F-28HD-441B"],
  ["Signature", "Valid C2PA-style signed record"],
];

const complianceItems = [
  ["Free public tool", "Upload or URL verification with no paid gate"],
  ["API access", "Documented endpoint for automated verification"],
  ["Latent disclosure", "Provider, system version, timestamp, and unique ID"],
  ["Manifest option", "Creator-controlled audible or companion disclosure"],
  ["No retention", "Submitted content is processed ephemerally"],
  ["Feedback loop", "Opt-in feedback collection for detection quality"],
];

const tests = [
  ["MP3 compression", "Pass"],
  ["Resampling", "Pass"],
  ["Peak normalization", "Pass"],
  ["Light trimming", "Review"],
  ["Background noise", "Pass"],
];

export function AudiomarkPortal() {
  const [mode, setMode] = useState<VerificationMode>("file");
  const [fileName, setFileName] = useState("");
  const [url, setUrl] = useState("");
  const [isChecking, setIsChecking] = useState(false);
  const [checked, setChecked] = useState(false);
  const [manifestEnabled, setManifestEnabled] = useState(true);
  const [feedbackSent, setFeedbackSent] = useState(false);

  const targetLabel = useMemo(() => {
    if (mode === "file") {
      return fileName || "demo_track.wav";
    }

    return url || "https://example.com/audio/demo-track.mp3";
  }, [fileName, mode, url]);

  function handleVerify(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsChecking(true);
    setChecked(false);

    window.setTimeout(() => {
      setIsChecking(false);
      setChecked(true);
    }, 800);
  }

  return (
    <main className="min-h-screen bg-white text-slate-950">
      <section className="hero-shell">
        <nav className="topbar" aria-label="Primary">
          <a className="brand" href="#verify" aria-label="Audiomark AI home">
            <span className="brand-mark">A</span>
            <span>Audiomark AI</span>
          </a>
          <div className="nav-actions">
            <a href="#api">API</a>
            <a href="#compliance">Compliance</a>
            <a className="nav-button" href="#verify">
              Verify
            </a>
          </div>
        </nav>

        <div className="hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">California AI Transparency Act MVP</p>
            <h1>Public audio provenance verification for GenAI providers.</h1>
            <p className="hero-text">
              A sleek verification portal and API for detecting latent audio
              disclosures, returning system provenance, and preserving a
              privacy-first no-retention posture.
            </p>
            <div className="hero-metrics" aria-label="Compliance highlights">
              <div>
                <strong>$0</strong>
                <span>public detection</span>
              </div>
              <div>
                <strong>0s</strong>
                <span>content retention</span>
              </div>
              <div>
                <strong>96h</strong>
                <span>license response clock</span>
              </div>
            </div>
          </div>

          <section className="verify-panel" id="verify" aria-labelledby="verify-title">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Detection Portal</p>
                <h2 id="verify-title">Verify audio provenance</h2>
              </div>
              <span className="status-pill">No content retained</span>
            </div>

            <div className="segmented" role="tablist" aria-label="Verification mode">
              <button
                className={mode === "file" ? "active" : ""}
                type="button"
                onClick={() => setMode("file")}
                aria-selected={mode === "file"}
              >
                File
              </button>
              <button
                className={mode === "url" ? "active" : ""}
                type="button"
                onClick={() => setMode("url")}
                aria-selected={mode === "url"}
              >
                URL
              </button>
            </div>

            <form className="verify-form" onSubmit={handleVerify}>
              {mode === "file" ? (
                <label className="drop-zone">
                  <span className="drop-icon">+</span>
                  <span className="drop-title">
                    {fileName || "Select an audio file"}
                  </span>
                  <span className="drop-subtitle">WAV, MP3, M4A up to 250 MB</span>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(event) =>
                      setFileName(event.target.files?.[0]?.name || "")
                    }
                  />
                </label>
              ) : (
                <label className="url-field">
                  <span>Audio URL</span>
                  <input
                    value={url}
                    onChange={(event) => setUrl(event.target.value)}
                    placeholder="https://example.com/audio.mp3"
                    type="url"
                  />
                </label>
              )}

              <button className="primary-button" type="submit">
                {isChecking ? "Checking..." : "Verify provenance"}
              </button>
            </form>

            <div className={`result-box ${checked ? "visible" : ""}`} aria-live="polite">
              <div>
                <span className="result-label">Watermark detected</span>
                <strong>{checked ? "97.8% confidence" : "Awaiting scan"}</strong>
                <p>{checked ? targetLabel : "Submit audio to display provenance."}</p>
              </div>
              <div className="confidence-ring" aria-hidden="true">
                {checked ? "98" : "--"}
              </div>
            </div>
          </section>
        </div>
      </section>

      <section className="content-band">
        <div className="section-heading">
          <p className="eyebrow">System Provenance</p>
          <h2>Detected statutory payload</h2>
        </div>
        <div className="provenance-grid">
          {provenanceFields.map(([label, value]) => (
            <article className="info-card" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </article>
          ))}
        </div>
      </section>

      <section className="split-band" id="compliance">
        <div>
          <p className="eyebrow">Provider Console</p>
          <h2>Embed disclosures at generation time</h2>
          <p>
            The MVP SDK creates a signed record, embeds a latent audio marker,
            and offers a manifest disclosure toggle before export.
          </p>

          <div className="toggle-row">
            <div>
              <strong>Manifest disclosure</strong>
              <span>Clear AI-generated label for exported audio</span>
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

      <section className="content-band white-band" id="api">
        <div className="section-heading">
          <p className="eyebrow">Public API</p>
          <h2>One endpoint for automated checks</h2>
        </div>

        <div className="api-grid">
          <div className="code-panel">
            <div className="code-tabs">
              <span>POST /api/verify</span>
              <span>JSON</span>
            </div>
            <pre>{`{
  "detected": true,
  "confidence": 0.978,
  "system_provenance": {
    "provider": "Audiomark AI",
    "system": "VoxForge Studio",
    "version": "2.4.1",
    "created_at": "2026-08-16T22:42:00Z",
    "unique_id": "amk_7Q9F-28HD-441B",
    "signature_valid": true
  },
  "personal_provenance": null,
  "retention": "processed_in_memory"
}`}</pre>
          </div>

          <div className="test-panel">
            <h3>Robustness suite</h3>
            {tests.map(([name, status]) => (
              <div className="test-row" key={name}>
                <span>{name}</span>
                <strong className={status === "Pass" ? "pass" : "review"}>
                  {status}
                </strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="feedback-band">
        <div>
          <p className="eyebrow">Feedback Loop</p>
          <h2>Improve detection without collecting unnecessary data</h2>
        </div>
        <form
          className="feedback-form"
          onSubmit={(event) => {
            event.preventDefault();
            setFeedbackSent(true);
          }}
        >
          <label>
            Detection feedback
            <textarea placeholder="False positive, missed watermark, or file transformation details" />
          </label>
          <label className="checkbox-label">
            <input type="checkbox" />
            <span>Opt in to follow-up about this report</span>
          </label>
          <button className="secondary-button" type="submit">
            {feedbackSent ? "Feedback received" : "Send feedback"}
          </button>
        </form>
      </section>
    </main>
  );
}
