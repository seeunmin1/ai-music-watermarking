import PROVIDER_PROVENANCE from "../../shared/provider_provenance.json" with { type: "json" };

const PROFILES = PROVIDER_PROVENANCE.profiles;
const VENDOR_LABELS = PROVIDER_PROVENANCE.vendor_labels;
const PROVIDER_SIGNAL_IDS = ["google_synthid", "openai_synthid", "elevenlabs_watermark", "suno_c2pa"];

function textIncludesAny(value, terms) {
  const text = String(value || "").toLowerCase();
  return terms.some((term) => text.includes(term));
}

function collectManifestText(manifest, fields) {
  if (!manifest && !fields) return "";
  return JSON.stringify({ manifest, fields }).toLowerCase();
}

function claimLevel(status) {
  if (status === "verified") return "verified";
  if (status === "probable") return "probable";
  return "unknown";
}

function manifestSignal(meta) {
  if (!meta?.manifest) return null;
  const fields = meta.fields || {};
  return {
    id: "c2pa_manifest",
    label: PROFILES.c2pa_manifest.label,
    category: "Verified provenance",
    status: "verified",
    claimLevel: "verified",
    confidence: 92,
    provider: fields.provider || null,
    system: fields.system || null,
    contentId: fields.uid || fields.unique_id || null,
    createdAt: fields.created || null,
    detail: "Embedded C2PA-style disclosure manifest parsed",
  };
}

function providerPayloadSignals(meta) {
  const text = collectManifestText(meta?.manifest, meta?.fields);
  const hints = new Set(meta?.vendorHints || []);
  const fields = meta?.fields || {};
  const out = [];

  for (const signalId of PROVIDER_SIGNAL_IDS) {
    const profile = PROFILES[signalId];
    const provider = profile.provider;
    const providerText = String(fields.provider || "").toLowerCase();
    const providerMatch = providerText && providerText.includes(provider.toLowerCase());
    const issuerMatch = textIncludesAny(text, profile.issuer_terms || []);
    const hintKey = provider.toLowerCase().replaceAll(" ", "");
    const hintOnly = (profile.hint_terms || []).some((term) => hints.has(term));

    if (text && textIncludesAny(text, profile.verified_terms || []) && (providerMatch || issuerMatch)) {
      out.push({
        id: signalId,
        label: profile.label,
        category: "Verified provenance",
        status: "verified",
        claimLevel: "verified",
        confidence: ["google_synthid", "openai_synthid"].includes(signalId) ? 95 : 93,
        provider,
        system: fields.system || profile.system,
        contentId: fields.uid || fields.unique_id || null,
        createdAt: fields.created || null,
        detail: `${provider} provenance payload recovered from metadata`,
      });
    } else if (hintOnly) {
      out.push({
        id: `metadata_vendor_hint:${hintKey}`,
        label: `${provider}-related metadata`,
        category: "Metadata hint",
        status: "hint",
        claimLevel: "unknown",
        confidence: 20,
        provider,
        system: "Unknown",
        contentId: null,
        createdAt: null,
        detail: `${provider} text was present, but no trusted payload or official result was recovered`,
      });
    }
  }

  return out;
}

function officialResultSignals(officialResults = []) {
  return officialResults.map((result) => {
    const provider = result.provider;
    const verified = !!result.verified;
    return {
      id: result.id || `official:${String(provider || "unknown").toLowerCase()}`,
      label: result.label || `${provider} official check`,
      category: verified ? "Verified provenance" : "Unsupported/unknown",
      status: verified ? "verified" : result.status || "not_configured",
      claimLevel: verified ? "verified" : "unknown",
      confidence: result.confidence || (verified ? 98 : 0),
      provider: verified ? provider : null,
      system: result.system || null,
      contentId: result.contentId || null,
      createdAt: result.createdAt || null,
      detail: result.detail || "Official provider verification adapter result",
      evidence: result.evidence || null,
    };
  });
}

function audiomarkSignal(wm, rec) {
  if (!wm?.found) return null;
  const registered = !!rec;
  return {
    id: "audiomark_ss_v1",
    label: PROFILES.audiomark_ss_v1.label,
    category: registered ? "Verified provenance" : "Detected watermark",
    status: registered ? "verified" : "detected",
    claimLevel: registered ? "verified" : "unknown",
    confidence: Number(wm.confidence || 0),
    provider: registered ? rec.provider : "Unknown",
    system: registered ? `${rec.system} v${rec.version}` : "Unknown",
    contentId: registered ? rec.uid : `record 0x${Number(wm.recordId || 0).toString(16).padStart(6, "0")}`,
    createdAt: registered ? rec.timestamp : null,
    detail: registered
      ? "Latent watermark matched a local registry record"
      : "Latent Audiomark watermark decoded without a registry match",
    recordStatus: rec?.status || null,
  };
}

function classifierSignal(classifierResult) {
  if (!classifierResult || classifierResult.status !== "probable") return null;
  const top = (classifierResult.candidates || [])[0] || {};
  const confidence = Math.round(Number(top.confidence || 0) * 100);
  if (!top.provider || confidence <= 0) return null;
  return {
    id: "local_attribution_classifier",
    label: PROFILES.local_attribution_classifier.label,
    category: "Probable attribution",
    status: "probable",
    claimLevel: "probable",
    confidence,
    provider: top.provider,
    system: top.system || "Unknown",
    contentId: null,
    createdAt: null,
    detail: classifierResult.detail || "Local ML attribution model result; not verified provenance",
  };
}

function vendorHintSignals(meta) {
  const providerHints = new Set(["google", "gemini", "synthid", "lyria", "openai", "chatgpt", "sora", "elevenlabs", "suno"]);
  return (meta?.vendorHints || [])
    .filter((v) => !providerHints.has(v))
    .map((v) => ({
      id: `metadata_vendor_hint:${v}`,
      label: VENDOR_LABELS[v] || v,
      category: "Metadata hint",
      status: "hint",
      claimLevel: "unknown",
      confidence: 15,
      provider: VENDOR_LABELS[v] || v,
      system: "Unknown",
      contentId: null,
      createdAt: null,
      detail: "Vendor string appeared in metadata bytes; not a verified provenance payload",
    }));
}

function wmarExperimentalSignal() {
  const profile = PROFILES.wmar_clustered_token;
  return {
    id: "wmar_clustered_token",
    label: "WMAR clustered-token detector",
    category: "Unsupported/unknown",
    status: "unsupported",
    claimLevel: "unknown",
    confidence: 0,
    provider: null,
    system: null,
    contentId: null,
    createdAt: null,
    detail: `Experimental backend hook only; upstream ${profile.upstream}`,
  };
}

function pickPrimary(signals) {
  return signals.find((s) => s.status === "verified" && s.id.startsWith("official:"))
    || PROVIDER_SIGNAL_IDS.map((id) => signals.find((s) => s.id === id && s.status === "verified")).find(Boolean)
    || signals.find((s) => s.id === "audiomark_ss_v1" && s.status === "verified")
    || signals.find((s) => s.id === "c2pa_manifest" && s.status === "verified")
    || signals.find((s) => s.status === "probable")
    || null;
}

function officialCheckStatus(signals, officialResults = []) {
  const configured = officialResults.filter((r) => r.configured);
  const verified = signals.filter((s) => s.status === "verified" && s.id.startsWith("official:"));
  return {
    status: verified.length ? "verified" : configured.length ? "checked_no_match" : "not_configured",
    configuredProviders: configured.map((r) => r.provider).filter(Boolean),
    checkedProviders: officialResults.map((r) => r.provider).filter(Boolean),
  };
}

export function analyzeProvenance({ wm, meta, registry, officialResults = [], classifierResult = null }) {
  const rec = wm?.found ? registry.find((r) => parseInt(r.idHex, 16) === wm.recordId) : null;
  const signals = [
    ...officialResultSignals(officialResults),
    manifestSignal(meta),
    ...providerPayloadSignals(meta),
    audiomarkSignal(wm, rec),
    classifierSignal(classifierResult),
    ...vendorHintSignals(meta),
    wmarExperimentalSignal(),
  ].filter(Boolean);

  const primary = pickPrimary(signals);
  const hasHints = signals.some((s) => s.status === "hint");
  const level = primary ? claimLevel(primary.status) : "unknown";
  const verdict = level === "verified"
    ? "ai_generated"
    : level === "probable"
      ? "probably_ai_generated"
      : hasHints
        ? "unknown_with_hints"
        : "unmarked_or_unknown";

  const providerCandidates = signals
    .filter((s) => s.provider && ["verified", "probable", "hint"].includes(s.status))
    .map((s) => ({
      provider: s.provider,
      system: s.system,
      confidence: s.confidence || 0,
      claimLevel: s.claimLevel || "unknown",
      resolvedVia: s.label,
      evidence: s.evidence || null,
    }));

  const limitations = [];
  if (!signals.some((s) => s.id === "google_synthid" && s.status === "verified")) {
    limitations.push("Google/Gemini is only verified when a trusted SynthID, manifest payload, or official adapter result is recovered.");
  }
  limitations.push("Classifier attribution is probable only and must not be treated as statutory provenance.");
  limitations.push("The WMAR clustered-token detector requires the experimental Python backend and model/codebook assets.");
  if (!primary && hasHints) limitations.push("Metadata hints are not statutory provenance on their own.");

  return {
    verdict,
    claimLevel: level,
    provider: primary?.provider || null,
    providerCandidates,
    system: primary?.system || null,
    contentId: primary?.contentId || null,
    createdAt: primary?.createdAt || null,
    signals,
    confidence: primary ? Math.max(1, Math.round(primary.confidence)) : 0,
    resolvedVia: primary?.label || null,
    officialCheckStatus: officialCheckStatus(signals, officialResults),
    limitations,
    record: rec || null,
  };
}

export { VENDOR_LABELS };
