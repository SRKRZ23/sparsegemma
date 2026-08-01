// draft_model.js — Gemma 3 270M-it as a speculative-decoding draft model.
//
// Unlike Gemma 4 E2B, this model takes `input_ids` directly (no PLE/inputs_embeds split — it does
// the embedding lookup internally via GatherBlockQuantized), so it's loaded and run as a plain
// onnxruntime-web session, no sparse-embed trick needed. Same 262,144-token vocab as the E2B
// target model, so draft tokens are directly usable for verification with no remapping.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort.webgpu.min.mjs";

const MODEL_URL = "https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX/resolve/main/onnx/model_q4f16.onnx";
const MODEL_DATA_URL = "https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX/resolve/main/onnx/model_q4f16.onnx_data";
const NUM_DRAFT_LAYERS = 18;
const KV_DIM = 256; // uniform across all 18 layers (unlike E2B's mixed 256/512)

let session = null;

export async function loadDraftModel(onProgress) {
  const graphResp = await fetch(MODEL_URL);
  const graphBuffer = new Uint8Array(await graphResp.arrayBuffer());

  const dataResp = await fetch(MODEL_DATA_URL);
  const total = parseInt(dataResp.headers.get("content-length") || "0", 10);
  const reader = dataResp.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received, total);
  }
  const dataBuffer = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { dataBuffer.set(c, off); off += c.length; }

  session = await ort.InferenceSession.create(graphBuffer, {
    executionProviders: ["webgpu", "wasm"],
    externalData: [{ path: "model_q4f16.onnx_data", data: dataBuffer }],
  });
  return session;
}

function emptyKV(batch = 1) {
  const feeds = {};
  for (let i = 0; i < NUM_DRAFT_LAYERS; i++) {
    feeds[`past_key_values.${i}.key`] = new ort.Tensor("float16", new Uint16Array(0), [batch, 1, 0, KV_DIM]);
    feeds[`past_key_values.${i}.value`] = new ort.Tensor("float16", new Uint16Array(0), [batch, 1, 0, KV_DIM]);
  }
  return feeds;
}

// Greedily draft up to `k` tokens starting from `contextIds` (the full sequence so far). Returns
// the proposed continuation token ids. Runs its own independent KV-cache from scratch each call —
// simpler and safe, at the cost of redoing the prefix forward pass each round. (An optimization —
// persisting the draft's own KV-cache across rounds the way the target does — is possible but was
// not necessary to get a correct, working speculative loop; the draft model is tiny, so a full
// from-scratch prefill is cheap relative to the target's per-token cost it's trying to amortize.)
export async function draftTokens(contextIds, k, eosIds) {
  if (!session) throw new Error("draft model not loaded");
  let ids = [...contextIds];
  const proposed = [];
  let feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(ids.length).fill(1n)), [1, ids.length]),
    ...emptyKV(),
  };
  for (let step = 0; step < k; step++) {
    const output = await session.run(feeds);
    const logits = output.logits.data; // [1, seqLen, vocab] but ONNX Runtime flattens; we only need the LAST position
    const vocab = output.logits.dims[2];
    const lastPosOffset = (output.logits.dims[1] - 1) * vocab;
    let bestIdx = 0, bestVal = -Infinity;
    for (let v = 0; v < vocab; v++) {
      const val = logits[lastPosOffset + v];
      if (val > bestVal) { bestVal = val; bestIdx = v; }
    }
    if (eosIds.has(bestIdx)) break;
    proposed.push(bestIdx);
    ids.push(bestIdx);

    const nextKV = {};
    for (let i = 0; i < NUM_DRAFT_LAYERS; i++) {
      nextKV[`past_key_values.${i}.key`] = output[`present.${i}.key`];
      nextKV[`past_key_values.${i}.value`] = output[`present.${i}.value`];
    }
    feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(bestIdx)]), [1, 1]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(ids.length).fill(1n)), [1, ids.length]),
      ...nextKV,
    };
  }
  return proposed;
}
