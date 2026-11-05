import { NBITS } from "./dsp.js";

/* Browser localStorage — MVP-scoped to this device/browser. */
export function loadState(key, fallback) {
  try {
    const raw = localStorage.getItem("audiomark:" + key);
    const value = raw ? JSON.parse(raw) : fallback;
    // Registry and audit history must be arrays, even when storage is valid JSON.
    if (Array.isArray(fallback) && (
      !Array.isArray(value)
      || value.some((item) => !item || typeof item !== "object" || Array.isArray(item))
    )) return fallback;
    return value;
  } catch {
    return fallback;
  }
}
export function saveState(key, value) {
  try {
    localStorage.setItem("audiomark:" + key, JSON.stringify(value));
  } catch {}
}

export function buildManifest(rec) {
  return {
    "@context": "https://c2pa.org/manifest (illustrative)",
    claim_generator: "AudiomarkAI/0.3.0-mvp",
    title: rec.fileName,
    assertions: [
      { label: "c2pa.actions", data: { actions: [{ action: "c2pa.created", digitalSourceType: "trainedAlgorithmicMedia" }] } },
      { label: "ai.statutory_disclosure.ca_sb942",
        data: { provider: rec.provider, system: rec.system, system_version: rec.version, created: rec.timestamp, unique_id: rec.uid } },
      { label: "audiomark.latent_watermark",
        data: { scheme: "audiomark-ss/v1", record_id: rec.idHex, bit_length: NBITS, redundancy: rec.repetitions } },
    ],
    content_bindings: { alg: "sha256", source_asset: rec.sourceHash },
    signature: { alg: "ed25519 (simulated in MVP)", value: "am1_" + rec.uid.replaceAll("-", "").slice(0, 40) },
  };
}
