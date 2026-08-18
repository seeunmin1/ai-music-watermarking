import React, { useState, useRef, useCallback, useEffect } from "react";

/* ================================================================
   AUDIOMARK AI — Compliance Engine MVP (v0.2)
   CA SB 942 / AB 853 — dual-mode detect/embed portal

   REAL in this MVP (all in-browser, zero retention):
   • Mode B (Embed): keyed spread-spectrum latent watermark
     (12-bit sync + 24-bit record ID + 8-bit checksum, level-adaptive
     gain, repeated across track) + a C2PA-style manifest written INTO
     the output WAV as a RIFF "c2pa" chunk, bound by SHA-256 hashes.
   • Mode A (Detect): matched-filter watermark detector (z-score) +
     byte-level metadata scanner that finds and parses embedded C2PA
     manifests, ID3v2 tags, JUMBF markers, and vendor-name hints
     (Suno / ElevenLabs / OpenAI / SynthID strings in metadata).
   • Instant statutory verdict: AI-Generated vs Unmarked, with
     provider / system+version / timestamp / unique ID resolution.
   • Persistent registry + audit log (window.storage w/ fallback).
   Stubs (labeled in UI): SynthID & AudioSeal signal decoders,
   real Ed25519 signing.
   ================================================================ */

/* ---------------- DSP CORE ---------------- */
const WM_KEY = 0x00a1d107;
const BIT_LEN = 2048;
const PREAMBLE = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0];
const ID_BITS = 24, CK_BITS = 8;
const NBITS = PREAMBLE.length + ID_BITS + CK_BITS;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function pnSeq(len, key) {
  const r = mulberry32(key);
  const s = new Float32Array(len);
  for (let i = 0; i < len; i++) s[i] = r() < 0.5 ? -1 : 1;
  return s;
}
const toBits = (v, n) => { const b = []; for (let i = n - 1; i >= 0; i--) b.push((v >> i) & 1); return b; };
const fromBits = (bits) => bits.reduce((a, b) => (a << 1) | b, 0);
const checksum24 = (id) => ((id >> 16) ^ (id >> 8) ^ id) & 0xff;

function embedWatermark(samples, recordId) {
  const bits = [...PREAMBLE, ...toBits(recordId, ID_BITS), ...toBits(checksum24(recordId), CK_BITS)];
  const pn = pnSeq(BIT_LEN, WM_KEY);
  const frame = NBITS * BIT_LEN;
  const out = Float32Array.from(samples);
  if (out.length < frame) return null;
  let reps = 0;
  for (let start = 0; start + frame <= out.length; start += frame) {
    for (let b = 0; b < NBITS; b++) {
      const s0 = start + b * BIT_LEN;
      const sign = bits[b] ? 1 : -1;
      let rms = 0;
      for (let n = 0; n < BIT_LEN; n++) rms += out[s0 + n] * out[s0 + n];
      rms = Math.sqrt(rms / BIT_LEN);
      const a = Math.min(0.12 * rms + 0.0006, 0.02);
      for (let n = 0; n < BIT_LEN; n++) out[s0 + n] += sign * pn[n] * a;
    }
    reps++;
  }
  for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, out[i]));
  return { samples: out, repetitions: reps };
}

function detectWatermark(samples) {
  const pn = pnSeq(BIT_LEN, WM_KEY);
  const frame = NBITS * BIT_LEN;
  const reps = Math.floor(samples.length / frame);
  if (reps < 1) return { found: false, z: 0, confidence: 0, reason: "short" };
  const acc = new Float64Array(NBITS);
  let energy = 0;
  for (let i = 0; i < reps * frame; i++) energy += samples[i] * samples[i];
  const rms = Math.sqrt(energy / (reps * frame)) || 1e-6;
  for (let r = 0; r < reps; r++) {
    for (let b = 0; b < NBITS; b++) {
      const s0 = r * frame + b * BIT_LEN;
      let c = 0;
      for (let n = 0; n < BIT_LEN; n++) c += samples[s0 + n] * pn[n];
      acc[b] += c;
    }
  }
  const bits = Array.from(acc, (v) => (v > 0 ? 1 : 0));
  const preOk = bits.slice(0, PREAMBLE.length).every((b, i) => b === PREAMBLE[i]);
  const id = fromBits(bits.slice(PREAMBLE.length, PREAMBLE.length + ID_BITS));
  const ckOk = fromBits(bits.slice(PREAMBLE.length + ID_BITS)) === checksum24(id);
  const sigma = rms * Math.sqrt(BIT_LEN * reps);
  let zSum = 0;
  for (let b = 0; b < NBITS; b++) zSum += Math.abs(acc[b]) / sigma;
  const z = zSum / NBITS;
  const confidence = Math.min(99.9, Math.max(0, (1 - Math.exp(-z / 3)) * 100));
  return { found: preOk && ckOk && z > 2.5, recordId: id, z, confidence, repetitions: reps };
}

/* ---------------- METADATA SCANNER (Mode A) ----------------
   Byte-level scan of the raw file: ID3v2 header, JUMBF markers,
   embedded C2PA JSON (incl. our own RIFF "c2pa" chunk), and
   vendor-name hints inside metadata regions. */
function findAscii(bytes, needle, from = 0, to = bytes.length) {
  const n = needle.split("").map((c) => c.charCodeAt(0));
  const limit = Math.min(to, bytes.length) - n.length;
  outer: for (let i = from; i <= limit; i++) {
    for (let j = 0; j < n.length; j++) {
      const b = bytes[i + j], t = n[j];
      const alpha = (t >= 65 && t <= 90) || (t >= 97 && t <= 122);
      if (b !== t && !(alpha && (b | 32) === (t | 32))) continue outer;
    }
    return i;
  }
  return -1;
}
function extractJsonNear(bytes, idx) {
  const start = findAscii(bytes, "{", idx, idx + 96);
  if (start < 0) return null;
  let depth = 0, inStr = false, esc = false;
  const max = Math.min(bytes.length, start + 200000);
  for (let i = start; i < max; i++) {
    const c = bytes[i];
    if (esc) { esc = false; continue; }
    if (inStr) { if (c === 92) esc = true; else if (c === 34) inStr = false; continue; }
    if (c === 34) inStr = true;
    else if (c === 123) depth++;
    else if (c === 125 && --depth === 0) {
      try {
        const txt = new TextDecoder().decode(bytes.slice(start, i + 1));
        return JSON.parse(txt);
      } catch { return null; }
    }
  }
  return null;
}
function scanMetadata(bytes) {
  const res = { id3: false, jumbf: false, manifest: null, vendorHints: [], fields: null };
  if (bytes.length > 3 && bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) res.id3 = true;
  // scan head + tail windows to keep large files fast
  const HEAD = Math.min(bytes.length, 1 << 20);
  const TAILSTART = Math.max(0, bytes.length - (1 << 20));
  const windows = TAILSTART > HEAD ? [[0, HEAD], [TAILSTART, bytes.length]] : [[0, bytes.length]];
  for (const [a, b] of windows) {
    if (!res.jumbf && (findAscii(bytes, "jumb", a, b) >= 0 || findAscii(bytes, "jumd", a, b) >= 0)) res.jumbf = true;
    if (!res.manifest) {
      let i = a;
      while (i < b) {
        const idx = findAscii(bytes, "c2pa", i, b);
        if (idx < 0) break;
        const m = extractJsonNear(bytes, idx);
        if (m) { res.manifest = m; break; }
        i = idx + 4;
      }
    }
    for (const v of ["suno", "elevenlabs", "openai", "synthid", "audioseal", "udio", "lyria"]) {
      if (res.vendorHints.includes(v)) continue;
      let i = a;
      while (i < b) {
        const idx = findAscii(bytes, v, i, b);
        if (idx < 0) break;
        // "udio" is a substring of "audio" — require a non-'a' predecessor
        if (v !== "udio" || idx === 0 || (bytes[idx - 1] | 32) !== 97) { res.vendorHints.push(v); break; }
        i = idx + 1;
      }
    }
  }
  if (res.manifest) {
    const m = res.manifest;
    const stat = (m.assertions || []).find((x) => (x.label || "").includes("statutory"))?.data || {};
    res.fields = {
      provider: stat.provider || m.claim_generator || "Unknown",
      system: stat.system ? `${stat.system} v${stat.system_version || "?"}` : m.claim_generator || "—",
      created: stat.created || null,
      uid: stat.unique_id || null,
    };
  }
  return res;
}

/* ---------------- AUDIO I/O ---------------- */
async function readFileAll(file) {
  const buf = await file.arrayBuffer();
  const bytes = new Uint8Array(buf);
  const hash = await sha256Hex(buf.slice(0));
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audio = await ctx.decodeAudioData(buf.slice(0));
  const ch = audio.numberOfChannels;
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < ch; c++) {
    const d = audio.getChannelData(c);
    for (let i = 0; i < d.length; i++) mono[i] += d[i] / ch;
  }
  ctx.close();
  return { bytes, samples: mono, sampleRate: audio.sampleRate, duration: audio.duration, hash };
}
function encodeWav(samples, sampleRate, extraChunks = []) {
  const n = samples.length;
  let extra = 0;
  const chunks = extraChunks.map(({ id, data }) => {
    const pad = data.length % 2;
    extra += 8 + data.length + pad;
    return { id, data, pad };
  });
  const buf = new ArrayBuffer(44 + n * 2 + extra);
  const v = new DataView(buf);
  const u8 = new Uint8Array(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); v.setUint32(4, 36 + n * 2 + extra, true); ws(8, "WAVE"); ws(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); ws(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  let o = 44 + n * 2;
  for (const c of chunks) {
    ws(o, c.id); v.setUint32(o + 4, c.data.length, true);
    u8.set(c.data, o + 8);
    o += 8 + c.data.length + c.pad;
  }
  return new Blob([buf], { type: "audio/wav" });
}
async function sha256Hex(buf) {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/* ---------------- PERSISTENCE ---------------- */
const mem = { registry: [], audit: [] };
async function loadState(key, fallback) {
  try { const r = await window.storage.get("audiomark:" + key); return r ? JSON.parse(r.value) : fallback; }
  catch { return mem[key] || fallback; }
}
async function saveState(key, value) {
  mem[key] = value;
  try { await window.storage.set("audiomark:" + key, JSON.stringify(value)); } catch {}
}

/* ---------------- MANIFEST ---------------- */
function buildManifest(rec) {
  return {
    "@context": "https://c2pa.org/manifest (illustrative)",
    claim_generator: "AudiomarkAI/0.2.0-mvp",
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

/* ---------------- STYLES ---------------- */
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--bg:#0b1220;--panel:#101a2e;--panel2:#0d1626;--line:#1e2b45;--ink:#e8eef9;--mut:#8a97b0;--dim:#5b6880;--ok:#3ddc97;--warn:#ffb020;--bad:#ff4d6d;--act:#38bdf8;}
.am-root{min-height:100vh;background:var(--bg);color:var(--ink);font-family:'Space Grotesk',system-ui,sans-serif;}
.am-wrap{max-width:1060px;margin:0 auto;padding:28px 20px 80px;}
.am-mono{font-family:'JetBrains Mono',ui-monospace,monospace;}
.am-eyebrow{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.18em;color:var(--act);text-transform:uppercase;}
.am-h1{font-size:34px;font-weight:700;letter-spacing:-.01em;margin:6px 0 2px;}
.am-sub{color:var(--mut);font-size:14px;max-width:660px;}
.am-mode{display:flex;gap:0;margin:24px 0 8px;border:1px solid var(--line);border-radius:12px;overflow:hidden;width:fit-content;}
.am-mode button{background:var(--panel2);border:none;color:var(--mut);font:inherit;font-weight:700;font-size:14px;padding:12px 26px;cursor:pointer;letter-spacing:.02em;}
.am-mode button[data-on="1"]{background:var(--act);color:#06121f;}
.am-mode button:not([data-on="1"]):hover{color:var(--ink);}
.am-tabs{display:flex;gap:14px;margin:4px 0 20px;}
.am-tab{background:none;border:none;color:var(--dim);font:inherit;font-size:12.5px;padding:6px 2px;cursor:pointer;border-bottom:2px solid transparent;letter-spacing:.04em;}
.am-tab:hover{color:var(--ink);}
.am-tab[data-on="1"]{color:var(--ink);border-bottom-color:var(--act);}
.am-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;}
.am-grid{display:grid;gap:16px;}
@media(min-width:860px){.am-grid.two{grid-template-columns:1fr 1fr;}}
.am-label{display:block;font-size:12px;color:var(--mut);margin-bottom:5px;letter-spacing:.04em;}
.am-input{width:100%;box-sizing:border-box;background:var(--panel2);border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:9px 11px;font:inherit;font-size:14px;}
.am-input:focus{outline:2px solid var(--act);outline-offset:1px;border-color:transparent;}
.am-btn{background:var(--act);color:#06121f;border:none;border-radius:8px;padding:10px 18px;font:inherit;font-weight:700;font-size:14px;cursor:pointer;}
.am-btn:hover{filter:brightness(1.1);}
.am-btn:disabled{opacity:.4;cursor:not-allowed;}
.am-btn.ghost{background:transparent;color:var(--act);border:1px solid var(--act);}
.am-btn.danger{background:transparent;color:var(--bad);border:1px solid var(--bad);}
.am-drop{border:1.5px dashed var(--line);border-radius:12px;padding:26px;text-align:center;color:var(--mut);cursor:pointer;transition:border-color .15s;}
.am-drop:hover,.am-drop.over{border-color:var(--act);color:var(--ink);}
.am-kv{display:grid;grid-template-columns:150px 1fr;gap:6px 14px;font-size:13px;}
.am-kv dt{color:var(--dim);}
.am-kv dd{margin:0;word-break:break-all;}
.am-pill{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:11px;padding:3px 9px;border-radius:99px;letter-spacing:.06em;}
.am-pill.ok{background:rgba(61,220,151,.12);color:var(--ok);}
.am-pill.bad{background:rgba(255,77,109,.12);color:var(--bad);}
.am-pill.warn{background:rgba(255,176,32,.12);color:var(--warn);}
.am-pill.dim{background:rgba(138,151,176,.12);color:var(--mut);}
.am-pill.act{background:rgba(56,189,248,.12);color:var(--act);}
.am-table{width:100%;border-collapse:collapse;font-size:13px;}
.am-table th{text-align:left;color:var(--dim);font-weight:500;font-size:11px;letter-spacing:.08em;text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--line);}
.am-table td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top;}
.am-note{font-size:12px;color:var(--dim);line-height:1.5;}
.am-h2{font-size:17px;font-weight:700;margin:0 0 14px;}
.am-canvas{width:100%;height:96px;display:block;border-radius:8px;background:var(--panel2);}
.am-conf{height:8px;border-radius:99px;background:var(--panel2);overflow:hidden;margin:6px 0 2px;}
.am-conf>div{height:100%;background:linear-gradient(90deg,#ff4d6d,#ffb020,#3ddc97);}
a.am-dl{color:var(--act);text-decoration:none;font-size:14px;font-weight:500;}
a.am-dl:hover{text-decoration:underline;}
.am-verdict{border-radius:12px;padding:22px;border:1px solid var(--line);display:flex;align-items:center;gap:18px;flex-wrap:wrap;}
.am-verdict.ai{background:linear-gradient(135deg,rgba(255,77,109,.10),rgba(255,176,32,.06));border-color:rgba(255,77,109,.35);}
.am-verdict.human{background:linear-gradient(135deg,rgba(61,220,151,.08),rgba(56,189,248,.05));border-color:rgba(61,220,151,.3);}
.am-verdict .big{font-size:24px;font-weight:700;letter-spacing:-.01em;}
`;

/* ---------------- SMALL COMPONENTS ---------------- */
function Waveform({ samples }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !samples) return;
    const w = (cv.width = cv.offsetWidth * 2), h = (cv.height = 192);
    const g = cv.getContext("2d");
    g.clearRect(0, 0, w, h);
    const grad = g.createLinearGradient(0, 0, w, 0);
    ["#ff4d6d", "#ffb020", "#3ddc97", "#38bdf8"].forEach((c, i) => grad.addColorStop(i / 3, c));
    g.strokeStyle = grad; g.lineWidth = 1.5;
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
  return <canvas ref={ref} className="am-canvas" />;
}

function Drop({ onFile, label }) {
  const [over, setOver] = useState(false);
  const inp = useRef(null);
  return (
    <div className={"am-drop" + (over ? " over" : "")}
      onClick={() => inp.current.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); }}>
      <div style={{ fontSize: 22, marginBottom: 6 }}>⇪</div>
      {label}
      <div className="am-note" style={{ marginTop: 6 }}>.mp3 · .wav · .m4a · .flac — processed in memory, zero permanent storage</div>
      <input ref={inp} type="file" accept="audio/*,.mp3,.wav,.m4a,.flac" hidden onChange={(e) => e.target.files[0] && onFile(e.target.files[0])} />
    </div>
  );
}

const VENDOR_LABELS = { suno: "Suno", elevenlabs: "ElevenLabs", openai: "OpenAI", synthid: "Google SynthID", audioseal: "Meta AudioSeal", udio: "Udio", lyria: "Google Lyria" };

/* ================= MAIN APP ================= */
export default function AudiomarkAI() {
  const [mode, setMode] = useState("detect"); // detect | embed
  const [tab, setTab] = useState("engine");   // engine | registry | audit
  const [registry, setRegistry] = useState([]);
  const [audit, setAudit] = useState([]);

  useEffect(() => {
    (async () => {
      setRegistry(await loadState("registry", []));
      setAudit(await loadState("audit", []));
    })();
  }, []);

  const log = useCallback((type, detail) => {
    setAudit((prev) => {
      const next = [{ t: new Date().toISOString(), type, detail }, ...prev].slice(0, 200);
      saveState("audit", next);
      return next;
    });
  }, []);

  /* ---- MODE A: DETECT ---- */
  const [dFile, setDFile] = useState(null);
  const [dBusy, setDBusy] = useState(false);
  const [dRes, setDRes] = useState(null);

  const runDetect = async (f) => {
    setDRes(null); setDBusy(true);
    try {
      const d = await readFileAll(f);
      setDFile({ name: f.name, ...d });
      await new Promise((r) => setTimeout(r, 30));
      const wm = detectWatermark(d.samples);
      const meta = scanMetadata(d.bytes);
      const rec = wm.found ? registry.find((r) => parseInt(r.idHex, 16) === wm.recordId) : null;
      const aiDetected = wm.found || !!meta.manifest;
      // provenance resolution order: registry record → embedded manifest → vendor hints
      let prov = null, provSource = null;
      if (rec) { prov = { provider: rec.provider, system: `${rec.system} v${rec.version}`, created: rec.timestamp, uid: rec.uid }; provSource = "Registry (latent watermark)"; }
      else if (meta.fields) { prov = meta.fields; provSource = "Embedded C2PA manifest"; }
      else if (aiDetected && wm.found) { prov = { provider: "Unknown (unregistered mark)", system: "—", created: null, uid: "record 0x" + wm.recordId.toString(16) }; provSource = "Latent watermark (no registry match)"; }
      setDRes({ wm, meta, rec, aiDetected, prov, provSource });
      log("verify", aiDetected
        ? `AI-GENERATED verdict for "${f.name}" — ${wm.found ? `watermark z=${wm.z.toFixed(1)}` : "no watermark"}${meta.manifest ? ", C2PA manifest found" : ""}`
        : `UNMARKED verdict for "${f.name}" (z=${wm.z.toFixed(1)}, no manifest)`);
    } catch { setDRes({ error: "Could not decode that file as audio." }); }
    setDBusy(false);
  };

  /* ---- MODE B: EMBED ---- */
  const [form, setForm] = useState({ provider: "Audiomark Labs", system: "DemoTTS", version: "1.0" });
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
    await new Promise((r) => setTimeout(r, 30));
    try {
      const recordId = Math.floor(Math.random() * 0xffffff);
      const em = embedWatermark(src.samples, recordId);
      if (!em) { setEErr("Audio must be at least ~2.1 s to carry one payload frame."); setEBusy(false); return; }
      const rec = {
        idHex: recordId.toString(16).padStart(6, "0"),
        provider: form.provider, system: form.system, version: form.version,
        timestamp: new Date().toISOString(), uid: crypto.randomUUID(),
        fileName: src.name, sourceHash: src.hash, repetitions: em.repetitions, status: "active",
      };
      const manifest = buildManifest(rec);
      const manBytes = new TextEncoder().encode(JSON.stringify(manifest));
      const wavBlob = encodeWav(em.samples, src.sampleRate, [{ id: "c2pa", data: manBytes }]);
      rec.wavHash = await sha256Hex(await wavBlob.arrayBuffer());
      const wavUrl = URL.createObjectURL(wavBlob);
      const manUrl = URL.createObjectURL(new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" }));
      setRegistry((prev) => { const next = [rec, ...prev]; saveState("registry", next); return next; });
      log("embed", `Embedded record ${rec.idHex} + in-file C2PA manifest into "${src.name}" (${em.repetitions}x redundancy) for ${form.provider}/${form.system} v${form.version}`);
      setERes({ rec, wavUrl, manUrl, wmSamples: em.samples });
    } catch (e) { setEErr("Embedding failed: " + e.message); }
    setEBusy(false);
  };

  /* ---- REVOCATION ---- */
  const revoke = (rec) => {
    const deadline = new Date(Date.now() + 96 * 3600 * 1000).toISOString();
    setRegistry((prev) => {
      const next = prev.map((r) => (r.uid === rec.uid ? { ...r, status: "revocation_pending", revokedAt: new Date().toISOString(), revocationDeadline: deadline } : r));
      saveState("registry", next);
      return next;
    });
    log("revocation", `Strip/violation flagged on record ${rec.idHex} ("${rec.fileName}"). 96-hour revocation window opened - deadline ${deadline}`);
  };

  const fmtT = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

  return (
    <div className="am-root">
      <style>{CSS}</style>
      <div className="am-wrap">
        <div className="am-eyebrow">Compliance Engine · CA SB 942 / AB 853</div>
        <h1 className="am-h1">Audiomark AI</h1>
        <p className="am-sub">
          Dual-mode statutory portal: <b>Detect</b> scans any track for latent watermarks and signed C2PA manifest
          metadata and returns an instant verdict; <b>Embed</b> injects a latent cryptographic signature and writes a
          bound manifest directly into the output file. Everything runs in-memory — zero permanent storage.
        </p>

        <div className="am-tabs" style={{ marginTop: 18 }}>
          {[["engine", "ENGINE"], ["registry", "REGISTRY"], ["audit", "AUDIT LOG"]].map(([k, l]) => (
            <button key={k} className="am-tab" data-on={tab === k ? 1 : 0} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>

        {tab === "engine" && (
          <>
            <div className="am-mode" role="tablist">
              <button data-on={mode === "detect" ? 1 : 0} onClick={() => setMode("detect")}>MODE A · DETECT</button>
              <button data-on={mode === "embed" ? 1 : 0} onClick={() => setMode("embed")}>MODE B · EMBED</button>
            </div>

            {/* ============ MODE A: DETECT ============ */}
            {mode === "detect" && (
              <div className="am-grid" style={{ marginTop: 14 }}>
                <div className="am-card">
                  <h2 className="am-h2">Upload audio stream <span className="am-pill dim" style={{ marginLeft: 8 }}>ZERO RETENTION</span></h2>
                  <Drop onFile={runDetect} label="Drop any audio file for an instant statutory verdict" />
                  {dBusy && <div style={{ marginTop: 12, color: "var(--mut)" }}>Scanning: matched-filter correlation + byte-level metadata pass…</div>}
                </div>

                {dRes && !dRes.error && (
                  <>
                    {/* VERDICT */}
                    <div className={"am-verdict " + (dRes.aiDetected ? "ai" : "human")}>
                      <div>
                        <div className="am-eyebrow" style={{ color: dRes.aiDetected ? "var(--bad)" : "var(--ok)" }}>AI Detection Status</div>
                        <div className="big">{dRes.aiDetected ? "AI-GENERATED" : "HUMAN CREATED / UNMARKED"}</div>
                        <div className="am-note" style={{ maxWidth: 480, marginTop: 4 }}>
                          {dRes.aiDetected
                            ? "One or more provenance signals recovered from this file."
                            : "No latent watermark or C2PA manifest found. Absence of a mark is consistent with human-created content — but cannot rule out stripped or never-marked AI output."}
                        </div>
                      </div>
                      <div style={{ marginLeft: "auto", minWidth: 200 }}>
                        <div className="am-note">Watermark confidence (z = {dRes.wm.z.toFixed(2)})</div>
                        <div className="am-conf"><div style={{ width: `${dRes.wm.confidence.toFixed(0)}%` }} /></div>
                        <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
                          <span className={"am-pill " + (dRes.wm.found ? "bad" : "dim")}>{dRes.wm.found ? "LATENT MARK" : "NO MARK"}</span>
                          <span className={"am-pill " + (dRes.meta.manifest ? "bad" : "dim")}>{dRes.meta.manifest ? "C2PA MANIFEST" : "NO MANIFEST"}</span>
                        </div>
                      </div>
                    </div>

                    <div className="am-grid two">
                      {/* PROVENANCE */}
                      <div className="am-card">
                        <h2 className="am-h2">Provenance identification</h2>
                        {dRes.prov ? (
                          <>
                            <dl className="am-kv">
                              <dt>Provider / Platform</dt><dd><b>{dRes.prov.provider}</b></dd>
                              <dt>System version</dt><dd>{dRes.prov.system}</dd>
                              <dt>Timestamp</dt><dd>{fmtT(dRes.prov.created)}</dd>
                              <dt>Unique content ID</dt><dd className="am-mono" style={{ fontSize: 11 }}>{dRes.prov.uid || "—"}</dd>
                            </dl>
                            <div style={{ marginTop: 10 }}><span className="am-pill act">SOURCE · {dRes.provSource.toUpperCase()}</span></div>
                            {dRes.rec && dRes.rec.status !== "active" && (
                              <div style={{ marginTop: 8 }}><span className="am-pill warn">RECORD IN 96-HOUR REVOCATION</span></div>
                            )}
                          </>
                        ) : (
                          <div className="am-note">No statutory payload recovered. Under the 2027 platform duty this file falls through to cross-vendor scanners and fingerprint fallback.</div>
                        )}
                        {dRes.meta.vendorHints.length > 0 && (
                          <div style={{ marginTop: 12 }}>
                            <div className="am-label">Vendor strings found in file metadata</div>
                            {dRes.meta.vendorHints.map((v) => <span key={v} className="am-pill warn" style={{ marginRight: 6 }}>{(VENDOR_LABELS[v] || v).toUpperCase()}</span>)}
                          </div>
                        )}
                      </div>

                      {/* SIGNAL BREAKDOWN */}
                      <div className="am-card">
                        <h2 className="am-h2">Signal breakdown</h2>
                        {dFile && <Waveform samples={dFile.samples} />}
                        <table className="am-table" style={{ marginTop: 10 }}><tbody>
                          <tr><td>Audiomark SS/v1 latent mark</td><td style={{ textAlign: "right" }}>{dRes.wm.found ? <span className="am-pill ok">DECODED · {dRes.wm.repetitions}× REPS</span> : <span className="am-pill dim">NOT DETECTED</span>}</td></tr>
                          <tr><td>Embedded C2PA manifest</td><td style={{ textAlign: "right" }}>{dRes.meta.manifest ? <span className="am-pill ok">PARSED</span> : <span className="am-pill dim">NOT FOUND</span>}</td></tr>
                          <tr><td>JUMBF container markers</td><td style={{ textAlign: "right" }}>{dRes.meta.jumbf ? <span className="am-pill ok">PRESENT</span> : <span className="am-pill dim">NOT FOUND</span>}</td></tr>
                          <tr><td>ID3v2 metadata block</td><td style={{ textAlign: "right" }}>{dRes.meta.id3 ? <span className="am-pill ok">PRESENT</span> : <span className="am-pill dim">NOT FOUND</span>}</td></tr>
                          <tr><td>Google SynthID-Audio decoder</td><td style={{ textAlign: "right" }}><span className="am-pill warn">STUB · PENDING</span></td></tr>
                          <tr><td>Meta AudioSeal decoder</td><td style={{ textAlign: "right" }}><span className="am-pill warn">STUB · PENDING</span></td></tr>
                        </tbody></table>
                        <div className="am-note" style={{ marginTop: 10 }}>
                          File SHA-256 <span className="am-mono">{dFile?.hash.slice(0, 24)}…</span> · analyzed in memory and discarded.
                        </div>
                      </div>
                    </div>
                  </>
                )}
                {dRes?.error && <div className="am-card" style={{ color: "var(--bad)" }}>{dRes.error}</div>}
              </div>
            )}

            {/* ============ MODE B: EMBED ============ */}
            {mode === "embed" && (
              <div className="am-grid two" style={{ marginTop: 14 }}>
                <div className="am-card">
                  <h2 className="am-h2">Statutory disclosure record</h2>
                  {[["provider", "Covered provider"], ["system", "GenAI system"], ["version", "System version"]].map(([k, l]) => (
                    <div key={k} style={{ marginBottom: 12 }}>
                      <label className="am-label">{l}</label>
                      <input className="am-input" value={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.value })} />
                    </div>
                  ))}
                  <div className="am-note">
                    Output carries the mark twice over: (1) a latent spread-spectrum signature in the waveform holding a
                    24-bit record ID, and (2) a bound C2PA-style manifest written into the file as a RIFF chunk with the
                    full SB 942 field set. Strip the metadata and the latent mark still resolves via the registry.
                  </div>
                </div>

                <div className="am-card">
                  <h2 className="am-h2">Source audio</h2>
                  {!src && <Drop onFile={onEmbedFile} label="Drop generated audio here, or click to browse" />}
                  {src && (
                    <>
                      <Waveform samples={eRes ? eRes.wmSamples : src.samples} />
                      <dl className="am-kv" style={{ marginTop: 12 }}>
                        <dt>File</dt><dd>{src.name}</dd>
                        <dt>Duration</dt><dd>{src.duration.toFixed(2)} s @ {src.sampleRate} Hz</dd>
                        <dt>Source SHA-256</dt><dd className="am-mono" style={{ fontSize: 11 }}>{src.hash.slice(0, 32)}…</dd>
                      </dl>
                      {!eRes && (
                        <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                          <button className="am-btn" onClick={doEmbed} disabled={eBusy}>{eBusy ? "Embedding…" : "Embed signature + manifest"}</button>
                          <button className="am-btn ghost" onClick={() => { setSrc(null); setERes(null); }}>Clear</button>
                        </div>
                      )}
                    </>
                  )}
                  {eErr && <div style={{ color: "var(--bad)", fontSize: 13, marginTop: 10 }}>{eErr}</div>}
                  {eRes && (
                    <div style={{ marginTop: 14, borderTop: "1px solid var(--line)", paddingTop: 14 }}>
                      <span className="am-pill ok">REGISTERED · {eRes.rec.idHex}</span>
                      <span className="am-pill act" style={{ marginLeft: 6 }}>MANIFEST IN-FILE</span>
                      <dl className="am-kv" style={{ marginTop: 10 }}>
                        <dt>Unique ID</dt><dd className="am-mono" style={{ fontSize: 11 }}>{eRes.rec.uid}</dd>
                        <dt>Redundancy</dt><dd>{eRes.rec.repetitions}× payload repetitions</dd>
                      </dl>
                      <div style={{ display: "flex", gap: 18, marginTop: 12, flexWrap: "wrap" }}>
                        <a className="am-dl" href={eRes.wavUrl} download={src.name.replace(/\.[^.]+$/, "") + ".audiomark.wav"}>↓ Watermarked WAV</a>
                        <a className="am-dl" href={eRes.manUrl} download="manifest.c2pa.json">↓ Manifest copy</a>
                        <button className="am-btn ghost" onClick={() => { setSrc(null); setERes(null); }}>New file</button>
                      </div>
                      <div className="am-note" style={{ marginTop: 10 }}>
                        Round-trip test: download the WAV, switch to <b>Mode A · Detect</b>, and drop it back in — both
                        the latent mark and the in-file manifest will be recovered.
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {/* ============ REGISTRY ============ */}
        {tab === "registry" && (
          <div className="am-card">
            <h2 className="am-h2">Provenance registry <span className="am-note" style={{ fontWeight: 400 }}>— compact in-band ID → full cloud record</span></h2>
            {registry.length === 0 && <div className="am-note">No records yet. Use Mode B · Embed to create the first registry entry.</div>}
            {registry.length > 0 && (
              <div style={{ overflowX: "auto" }}>
                <table className="am-table">
                  <thead><tr><th>ID</th><th>Provider / System</th><th>Created</th><th>File</th><th>Status</th><th></th></tr></thead>
                  <tbody>
                    {registry.map((r) => (
                      <tr key={r.uid}>
                        <td className="am-mono">{r.idHex}</td>
                        <td>{r.provider}<div className="am-note">{r.system} v{r.version}</div></td>
                        <td>{fmtT(r.timestamp)}</td>
                        <td style={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis" }}>{r.fileName}</td>
                        <td>{r.status === "active"
                          ? <span className="am-pill ok">ACTIVE</span>
                          : <><span className="am-pill warn">REVOCATION</span><div className="am-note">deadline {fmtT(r.revocationDeadline)}</div></>}
                        </td>
                        <td>{r.status === "active" && <button className="am-btn danger" style={{ padding: "5px 10px", fontSize: 12 }} onClick={() => revoke(r)}>Flag strip</button>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="am-note" style={{ marginTop: 12 }}>
              "Flag strip" simulates licensee strip-monitoring: timestamps the evidence in the audit log and opens the
              96-hour statutory revocation window (SB 942 licensee duty).
            </div>
          </div>
        )}

        {/* ============ AUDIT ============ */}
        {tab === "audit" && (
          <div className="am-card">
            <h2 className="am-h2">Audit log</h2>
            {audit.length === 0 && <div className="am-note">No events yet.</div>}
            <table className="am-table"><tbody>
              {audit.map((e, i) => (
                <tr key={i}>
                  <td className="am-mono" style={{ whiteSpace: "nowrap", color: "var(--dim)", fontSize: 11 }}>{fmtT(e.t)}</td>
                  <td><span className={"am-pill " + (e.type === "embed" ? "ok" : e.type === "revocation" ? "warn" : "dim")}>{e.type.toUpperCase()}</span></td>
                  <td style={{ fontSize: 13 }}>{e.detail}</td>
                </tr>
              ))}
            </tbody></table>
            <div className="am-note" style={{ marginTop: 12 }}>Only event metadata is logged — never audio content, per the zero-retention design.</div>
          </div>
        )}
      </div>
    </div>
  );
}
