// sparse_embed.js — token-sparse embedding fetcher for Gemma 4 E2B's embed_tokens weights.
//
// The novel idea: embed_tokens_q4f16.onnx_data is a 1.59GB file, but its content is a plain
// per-token lookup table (each token id maps to a fixed, contiguous byte range — an ordinary
// row-major embedding matrix under Microsoft's `GatherBlockQuantized` block quantization).
// A short conversation only ever touches a few hundred distinct tokens out of the 262,144-word
// vocabulary. Instead of downloading the whole file, we HTTP-Range-fetch only the rows for
// tokens that actually appear — verified against real bytes from the model (see project notes:
// mean/std of the dequantized output matches a real trained embedding, not garbage).
//
// This is genuinely unbuilt elsewhere (checked: llama.cpp, Google LiteRT-LM, ONNX Runtime GenAI,
// Transformers.js all load the PLE/embedding tables in full) — see gemma_ple_lazy_loading notes.

const DATA_URL =
  "https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/embed_tokens_q4f16.onnx_data";

// Layout confirmed via `onnx.load(..., load_external_data=False)` on the small .onnx graph file —
// each initializer's external_data{offset,length} within the single .onnx_data blob.
const MAIN = {
  quant: { offset: 0, rowBytes: 768 }, // 262144 x 768 (packed 4-bit, 2 vals/byte -> 1536 dims)
  scales: { offset: 201326592, rowBytes: 96 }, // 262144 x 48 fp16 (block_size=32 -> 1536/32=48 groups)
  zp: { offset: 226492416, rowBytes: 24 }, // 262144 x 24 (packed 4-bit -> 48 zero-points)
};
const PLE = {
  quant: { offset: 232783872, rowBytes: 4480 }, // 262144 x 4480 (packed -> 8960 = 35*256 dims)
  scales: { offset: 1407188992, rowBytes: 560 }, // 262144 x 280 fp16 (8960/32=280 groups)
  zp: { offset: 1553989632, rowBytes: 140 }, // 262144 x 140 (packed -> 280 zero-points)
};
const BLOCK_SIZE = 32;
const MAIN_GLOBAL_SCALE = 39.25; // Mul constant after GatherBlockQuantized in the ONNX graph
const PLE_GLOBAL_SCALE = 16.0;
const MAIN_DIM = 1536;
const PLE_DIM_TOTAL = 8960; // 35 layers x 256
const NUM_LAYERS = 35;
const PLE_LAYER_DIM = 256;

// Persistent cross-session cache (biomimicry: "myelination"/immune-memory-cell framing — once a
// token's embedding is fetched once on this device, it never needs a network round-trip again,
// on this page load OR any future visit). IndexedDB, not just an in-memory Map, so it survives
// tab close/reload.
const DB_NAME = "pockettriage-embeddings";
const STORE = "tokens";
let dbPromise = null;
function openDB() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}
async function idbGet(tokenId) {
  try {
    const db = await openDB();
    return await new Promise((resolve) => {
      const tx = db.transaction(STORE, "readonly").objectStore(STORE).get(tokenId);
      tx.onsuccess = () => resolve(tx.result || null);
      tx.onerror = () => resolve(null);
    });
  } catch { return null; }
}
async function idbPut(tokenId, value) {
  try {
    const db = await openDB();
    db.transaction(STORE, "readwrite").objectStore(STORE).put(value, tokenId);
  } catch { /* best-effort cache; a write failure shouldn't break generation */ }
}

// Fetch a single byte range. Returns an ArrayBuffer.
//
// NOTE: cannot cache/reuse a resolved CDN URL across different ranges — verified via direct
// curl that HF's Xet CDN signs each redirect URL with a policy scoped to the EXACT Range header
// used to obtain it (`"ByteRange":{"ExpectedHeader":"bytes=0-0"}}` in the decoded policy JWT) —
// reusing that URL for a different range returns 403. Each fetch must go through the
// huggingface.co redirector fresh; the 2-hop cost per request is unavoidable with this CDN.
// Throughput comes from concurrency (see embedTokensSparse's worker pool), not URL reuse.
async function fetchRange(offset, length) {
  const start = offset;
  const end = offset + length - 1;
  const res = await fetch(DATA_URL, { headers: { Range: `bytes=${start}-${end}` } });
  if (res.status !== 206 && res.status !== 200) {
    throw new Error(`Range fetch failed: ${res.status} for bytes=${start}-${end}`);
  }
  return res.arrayBuffer();
}

function unpackNibbles(bytes) {
  // Low-nibble-first convention (byte&0xF = even index, byte>>4 = odd index) — matches
  // Microsoft's GatherBlockQuantized packing, verified by sane dequantized output statistics.
  const out = new Uint8Array(bytes.length * 2);
  for (let i = 0; i < bytes.length; i++) {
    out[i * 2] = bytes[i] & 0x0f;
    out[i * 2 + 1] = (bytes[i] >> 4) & 0x0f;
  }
  return out;
}

function dequantizeRow(quantBuf, scalesBuf, zpBuf, globalScale) {
  const quant = unpackNibbles(new Uint8Array(quantBuf));
  const scales = new Float32Array(new Uint16Array(scalesBuf).length);
  {
    // fp16 -> fp32 (no native Float16Array typed view for arithmetic; decode manually)
    const u16 = new Uint16Array(scalesBuf);
    for (let i = 0; i < u16.length; i++) scales[i] = fp16ToFloat32(u16[i]);
  }
  const zp = unpackNibbles(new Uint8Array(zpBuf));
  const out = new Float32Array(quant.length);
  // LUT-based dequant (T-MAC's core insight, arXiv:2407.00088): a 4-bit value has only 16
  // possible codes, but each group has BLOCK_SIZE=32 elements — so precomputing the 16 possible
  // dequantized values per group ONCE and looking them up is strictly cheaper (16 multiplies +
  // 32 lookups) than recomputing (value-zp)*scale for all 32 elements individually.
  const lut = new Float32Array(16);
  const numGroups = scales.length;
  for (let group = 0; group < numGroups; group++) {
    const s = scales[group] * globalScale;
    const z = zp[group];
    for (let v = 0; v < 16; v++) lut[v] = (v - z) * s;
    const start = group * BLOCK_SIZE;
    const end = Math.min(start + BLOCK_SIZE, quant.length);
    for (let i = start; i < end; i++) out[i] = lut[quant[i]];
  }
  return out;
}

function fp16ToFloat32(h) {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  let val;
  if (exp === 0) val = (frac / 1024) * Math.pow(2, -14);
  else if (exp === 0x1f) val = frac ? NaN : Infinity;
  else val = (1 + frac / 1024) * Math.pow(2, exp - 15);
  return sign ? -val : val;
}

// Fetch + dequantize the main-table and per-layer-table rows for one token id, in parallel.
// Checks the persistent IndexedDB cache first — a token fetched once (this session or a past
// one) never costs a network round-trip again.
async function fetchTokenEmbedding(tokenId) {
  const cached = await idbGet(tokenId);
  if (cached) return { mainEmb: new Float32Array(cached.mainEmb), perLayerFlat: new Float32Array(cached.perLayerFlat), fromCache: true };

  const [mq, ms, mz, pq, ps, pz] = await Promise.all([
    fetchRange(MAIN.quant.offset + tokenId * MAIN.quant.rowBytes, MAIN.quant.rowBytes),
    fetchRange(MAIN.scales.offset + tokenId * MAIN.scales.rowBytes, MAIN.scales.rowBytes),
    fetchRange(MAIN.zp.offset + tokenId * MAIN.zp.rowBytes, MAIN.zp.rowBytes),
    fetchRange(PLE.quant.offset + tokenId * PLE.quant.rowBytes, PLE.quant.rowBytes),
    fetchRange(PLE.scales.offset + tokenId * PLE.scales.rowBytes, PLE.scales.rowBytes),
    fetchRange(PLE.zp.offset + tokenId * PLE.zp.rowBytes, PLE.zp.rowBytes),
  ]);
  const mainEmb = dequantizeRow(mq, ms, mz, MAIN_GLOBAL_SCALE); // length 1536
  const perLayerFlat = dequantizeRow(pq, ps, pz, PLE_GLOBAL_SCALE); // length 8960 -> [35,256]
  idbPut(tokenId, { mainEmb: mainEmb.buffer, perLayerFlat: perLayerFlat.buffer }); // fire-and-forget
  return { mainEmb, perLayerFlat };
}

// Public API: given an array of token ids (as produced by the tokenizer), fetch/dequantize only
// the UNIQUE tokens present, then assemble the batch tensors the decoder expects.
export async function embedTokensSparse(tokenIds) {
  const unique = [...new Set(tokenIds)];
  const cache = new Map();
  const CONCURRENCY = 24; // stay well under typical browser per-host connection limits
  let idx = 0;
  let done = 0;
  const t0 = Date.now();
  async function worker() {
    while (idx < unique.length) {
      const t = unique[idx++];
      cache.set(t, await fetchTokenEmbedding(t));
      done++;
      if (done % 25 === 0 || done === unique.length) {
        console.log(`[sparse_embed] ${done}/${unique.length} tokens fetched, ${Date.now() - t0}ms elapsed`);
      }
    }
  }
  console.log(`[sparse_embed] starting fetch for ${unique.length} unique tokens (concurrency=${CONCURRENCY})`);
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, unique.length) }, worker));
  const seqLen = tokenIds.length;
  const inputsEmbeds = new Float32Array(seqLen * MAIN_DIM);
  const perLayerInputs = new Float32Array(seqLen * PLE_DIM_TOTAL);
  let cacheHits = 0;
  tokenIds.forEach((t, i) => {
    const { mainEmb, perLayerFlat } = cache.get(t);
    inputsEmbeds.set(mainEmb, i * MAIN_DIM);
    perLayerInputs.set(perLayerFlat, i * PLE_DIM_TOTAL);
  });
  for (const v of cache.values()) if (v.fromCache) cacheHits++;
  return {
    inputsEmbeds, // Float32Array, reshape to [1, seqLen, 1536]
    perLayerInputs, // Float32Array, reshape to [1, seqLen, 35, 256]
    bytesFetched: (unique.length - cacheHits) * (768 + 96 + 24 + 4480 + 560 + 140),
    uniqueTokens: unique.length,
    cacheHits,
  };
}

export const DIMS = { MAIN_DIM, PLE_DIM_TOTAL, NUM_LAYERS, PLE_LAYER_DIM };
