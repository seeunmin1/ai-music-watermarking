/* ================================================================
   Audiomark AI — watermarking DSP core + audio I/O + metadata scan.
   Keyed spread-spectrum latent watermark (12-bit sync + 24-bit
   record ID + 8-bit checksum, level-adaptive gain, repeated across
   the track) plus a byte-level scanner for embedded C2PA manifests,
   ID3v2 tags, JUMBF markers, and vendor-name hints. Mirrors cli/audiomark.py
   bit-for-bit so files embedded by one decode in the other.
   ================================================================ */

export const WM_KEY = 0x00a1d107;
export const BIT_LEN = 2048;
export const PREAMBLE = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0];
export const ID_BITS = 24, CK_BITS = 8;
export const NBITS = PREAMBLE.length + ID_BITS + CK_BITS;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function pnSeq(len, key) {
  const r = mulberry32(key);
  const s = new Float32Array(len);
  for (let i = 0; i < len; i++) s[i] = r() < 0.5 ? -1 : 1;
  return s;
}
const toBits = (v, n) => { const b = []; for (let i = n - 1; i >= 0; i--) b.push((v >> i) & 1); return b; };
const fromBits = (bits) => bits.reduce((a, b) => (a << 1) | b, 0);
const checksum24 = (id) => ((id >> 16) ^ (id >> 8) ^ id) & 0xff;

export function embedWatermark(samples, recordId) {
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

export function detectWatermark(samples) {
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

/* ---------------- METADATA SCANNER ----------------
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
export const VENDOR_LABELS = { suno: "Suno", elevenlabs: "ElevenLabs", openai: "OpenAI", chatgpt: "ChatGPT", sora: "Sora", synthid: "Google SynthID", google: "Google", gemini: "Google Gemini", audioseal: "Meta AudioSeal", udio: "Udio", lyria: "Google Lyria", "stability ai": "Stability AI", "stable audio": "Stable Audio", meta: "Meta", musicgen: "Meta MusicGen", adobe: "Adobe", firefly: "Adobe Firefly", microsoft: "Microsoft", copilot: "Microsoft Copilot", "azure ai": "Azure AI", bytedance: "ByteDance", tiktok: "TikTok", "resemble ai": "Resemble AI", playht: "PlayHT", "play.ht": "PlayHT", murf: "Murf", lovo: "Lovo", genny: "Lovo Genny", wellsaid: "WellSaid", speechify: "Speechify", descript: "Descript", overdub: "Descript Overdub", runway: "Runway", runwayml: "Runway" };

function isTokenByte(value) {
  const lower = value | 32;
  return (value >= 48 && value <= 57) || (lower >= 97 && lower <= 122);
}

function hasStandaloneAscii(bytes, text, from, to) {
  let i = from;
  while (i < to) {
    const idx = findAscii(bytes, text, i, to);
    if (idx < 0) return false;
    const beforeOk = idx === 0 || !isTokenByte(bytes[idx - 1]);
    const after = idx + text.length;
    const afterOk = after >= bytes.length || !isTokenByte(bytes[after]);
    if (beforeOk && afterOk) return true;
    i = idx + 1;
  }
  return false;
}

export function scanMetadata(bytes) {
  const res = { id3: false, jumbf: false, manifest: null, vendorHints: [], fields: null };
  if (bytes.length > 3 && bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) res.id3 = true;
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
    for (const v of Object.keys(VENDOR_LABELS)) {
      if (res.vendorHints.includes(v)) continue;
      if (hasStandaloneAscii(bytes, v, a, b)) res.vendorHints.push(v);
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
      unique_id: stat.unique_id || null,
    };
  }
  return res;
}

/* ---------------- AUDIO I/O ---------------- */
export async function readFileAll(file) {
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
export function encodeWav(samples, sampleRate, extraChunks = []) {
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
export async function sha256Hex(buf) {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
