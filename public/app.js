// PocketTriage — fully on-device health-navigation companion.
//
// NOVEL ARCHITECTURE: rather than using Transformers.js's high-level
// Gemma4ForConditionalGeneration (which eagerly downloads BOTH embed_tokens [1.59GB] and
// decoder_model_merged [1.52GB] — 3.1GB total), this loads ONLY the decoder session directly via
// onnxruntime-web, and supplies embeddings via sparse_embed.js — which HTTP-Range-fetches just the
// per-token rows actually used (a few KB per unique token) instead of the whole embed_tokens file.
// See sparse_embed.js for the reverse-engineered dequantization math (verified against real model
// bytes). This is, as far as we could research, not implemented anywhere else (llama.cpp, Google's
// own LiteRT-LM, ONNX Runtime GenAI, Transformers.js all load the full table).
import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort.webgpu.min.mjs";
import { embedTokensSparse, DIMS } from "./sparse_embed.js";
import { loadDraftModel, draftTokens } from "./draft_model.js";

const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";
const DECODER_URL =
  "https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx";
const DECODER_DATA_URL =
  "https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data";

window.addEventListener("error", (e) => console.error("[window.error]", e.message, e.error?.stack));
window.addEventListener("unhandledrejection", (e) => console.error("[unhandledrejection]", e.reason?.message || e.reason, e.reason?.stack));

// From inspecting the decoder's ONNX graph inputs directly: 15 attention layers, multi-query
// (1 KV head), most with head_dim=256 but layers 4, 9, 14 use head_dim=512.
const NUM_DECODER_LAYERS = 15;
const KV_DIM = (i) => ([4, 9, 14].includes(i) ? 512 : 256);

const statusEl = document.getElementById("status");
const loadfill = document.getElementById("loadfill");
const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

addMsg("sys", "Loading Gemma 4 decoder (sparse embeddings — only used tokens are fetched, not the full 1.59GB embedding table).");

const SYSTEM_PROMPT = `You are PocketTriage, a private, on-device health-navigation assistant. You are NOT a
doctor and you do NOT diagnose. Your job: (1) explain, in plain language, the general URGENCY category a
description of symptoms would typically fall under (similar to how ED triage systems like ESI/qSOFA reason
about urgency — routine / soon / urgent / emergency), (2) explain WHY in plain terms, (3) suggest 2-3
specific questions the person should ask a real clinician, and (4) always end by reminding them this is
educational only and to seek real medical care, calling emergency services immediately for anything
severe (chest pain, difficulty breathing, stroke signs, severe bleeding, loss of consciousness). Keep
answers concise (under 180 words). Never ask for or reference real identifying personal information.`;

let tokenizer, session;
let draftReady = false;
let history = [{ role: "system", content: SYSTEM_PROMPT }];

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
function float32ToFp16(f) {
  const floatView = new Float32Array(1);
  const int32View = new Uint32Array(floatView.buffer);
  floatView[0] = f;
  const x = int32View[0];
  const sign = (x >> 16) & 0x8000;
  let exp = ((x >> 23) & 0xff) - 127 + 15;
  let frac = x & 0x7fffff;
  if (exp <= 0) return sign; // flush to zero/subnormal-ish, good enough for zero-init tensors
  if (exp >= 0x1f) return sign | 0x7c00;
  return sign | (exp << 10) | (frac >> 13);
}

function emptyKV(batch = 1) {
  const feeds = {};
  for (let i = 0; i < NUM_DECODER_LAYERS; i++) {
    const dim = KV_DIM(i);
    feeds[`past_key_values.${i}.key`] = new ort.Tensor("float16", new Uint16Array(0), [batch, 1, 0, dim]);
    feeds[`past_key_values.${i}.value`] = new ort.Tensor("float16", new Uint16Array(0), [batch, 1, 0, dim]);
  }
  return feeds;
}

// Idea #5 + biomimicry #1/#6 (immune-memory / synaptic-pruning-by-usage framing): pre-warm the
// IndexedDB embedding cache with common health-navigation vocabulary at idle time, so the first
// real message pays less per-token network latency for ordinary words — only genuinely rare/
// domain-specific tokens still cost a fresh fetch.
const COMMON_VOCAB_SEED = [
  "I have been feeling anxious and tired for the past few days.",
  "There is a mild headache, some nausea, and dizziness since this morning.",
  "Sore throat, cough, mild fever, runny nose, body aches.",
  "Chest pain, shortness of breath, rapid heartbeat, sweating.",
  "Stomach pain, vomiting, diarrhea, loss of appetite, cramping.",
  "Rash on the skin, itching, swelling, redness, allergic reaction.",
  "Back pain, joint pain, muscle soreness after exercise, stiffness.",
  "Feeling dizzy, lightheaded, blurred vision, weakness, fatigue.",
  "What should I ask my doctor? Is this urgent or can it wait?",
  "Please explain in simple terms what this symptom usually means.",
].join(" ");

async function prewarmCommonVocab() {
  try {
    const { input_ids } = await tokenizer(COMMON_VOCAB_SEED, { return_tensor: false });
    const ids = Array.isArray(input_ids[0]) ? input_ids[0] : input_ids;
    const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 200));
    idle(async () => {
      const t0 = performance.now();
      const r = await embedTokensSparse(ids, { priority: "low" });
      console.log(`[prewarm] warmed ${r.uniqueTokens} common tokens (${r.cacheHits} already cached) in ${(performance.now() - t0).toFixed(0)}ms`);
    });
  } catch (err) {
    console.warn("[prewarm] skipped (non-fatal):", err?.message || err);
  }
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  const total = parseInt(resp.headers.get("content-length") || "0", 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let receivedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedBytes += value.length;
    onProgress(receivedBytes, total);
  }
  const buffer = new Uint8Array(receivedBytes);
  let off = 0;
  for (const c of chunks) { buffer.set(c, off); off += c.length; }
  return buffer;
}

async function init() {
  try {
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    ort.env.wasm.numThreads = 1; // cross-origin isolation unavailable on a plain static host

    // Two files: the small .onnx graph descriptor, and the ~1.5GB external weights blob it
    // references by filename (ORT needs both — this is the dense decoder, no sparsity trick
    // applies here the way it does for the embedding table; see sparse_embed.js).
    let graphBytes = 0, dataBytes = 0, dataTotal = 0;
    const modelBuffer = await fetchWithProgress(DECODER_URL, (n, t) => {
      graphBytes = n;
      statusEl.textContent = `loading decoder graph… ${(n / 1e6).toFixed(1)}MB`;
    });
    const externalDataBuffer = await fetchWithProgress(DECODER_DATA_URL, (n, t) => {
      dataBytes = n; dataTotal = t;
      const pct = t ? ((n / t) * 100).toFixed(1) : 0;
      loadfill.style.width = `${pct}%`;
      statusEl.textContent = `loading decoder weights… ${(n / 1e6).toFixed(0)}MB / ${(t / 1e6).toFixed(0)}MB`;
    });

    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["webgpu", "wasm"],
      externalData: [{ path: "decoder_model_merged_q4f16.onnx_data", data: externalDataBuffer }],
    });

    statusEl.textContent = "ready — decoder on-device (embeddings fetched sparsely, per-token)";
    statusEl.classList.remove("off");
    statusEl.classList.add("on");
    sendBtn.disabled = false;
    loadfill.style.width = "100%";
    addMsg("sys", "Ready. Embeddings for each token you use are fetched individually (a few KB each) instead of downloading the full 1.59GB embedding table.");

    prewarmCommonVocab(); // idle-time background prefetch, never blocks the UI

    // Load the speculative-decoding draft model (Gemma 3 270M-it, ~273MB) in the background —
    // the app is fully usable (plain greedy decode) without it; when it finishes, generation
    // switches to the faster draft-then-verify path automatically.
    loadDraftModel((n, t) => {
      if (t) statusEl.title = `loading speculative draft model… ${(n / 1e6).toFixed(0)}MB / ${(t / 1e6).toFixed(0)}MB`;
    }).then(() => {
      draftReady = true;
      window.draftReady = true; // exposed for automated testing
      addMsg("sys", "Speculative decoding ready (draft model: Gemma 3 270M) — generation should now be faster.");
    }).catch((err) => {
      console.warn("[draft] failed to load, falling back to plain greedy decode:", err?.message || err);
    });
  } catch (err) {
    statusEl.textContent = "failed to load (needs a WebGPU browser, e.g. recent Chrome/Edge)";
    addMsg("sys", `Load error: ${err?.message || err}`);
    console.error(err);
  }
}

function buildChatPrompt(msgs) {
  // Gemma chat template (text-only): <start_of_turn>role\n...content...<end_of_turn>\n
  let text = "";
  for (const m of msgs) {
    const role = m.role === "assistant" ? "model" : m.role;
    text += `<start_of_turn>${role}\n${m.content}<end_of_turn>\n`;
  }
  text += "<start_of_turn>model\n";
  return text;
}

async function generate(promptTokenIds, maxNewTokens = 220) {
  let pastKV = emptyKV();
  let allTokenIds = [...promptTokenIds];
  let generated = [];

  // --- first forward pass: full prompt ---
  let { inputsEmbeds, perLayerInputs, bytesFetched, uniqueTokens } = await embedTokensSparse(allTokenIds);
  console.log(`[sparse_embed] prompt: ${uniqueTokens} unique tokens, ${(bytesFetched / 1024).toFixed(1)}KB fetched (vs 1.59GB full table)`);
  let seqLen = allTokenIds.length;

  let feeds = {
    inputs_embeds: new ort.Tensor("float32", inputsEmbeds, [1, seqLen, DIMS.MAIN_DIM]),
    per_layer_inputs: new ort.Tensor("float32", perLayerInputs, [1, seqLen, DIMS.NUM_LAYERS, DIMS.PLE_LAYER_DIM]),
    attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(seqLen).fill(1n)), [1, seqLen]),
    position_ids: new ort.Tensor("int64", BigInt64Array.from(Array.from({ length: seqLen }, (_, i) => BigInt(i))), [1, seqLen]),
    num_logits_to_keep: new ort.Tensor("int64", BigInt64Array.from([1n]), []),
    ...pastKV,
  };

  const eosIds = new Set([1, 106, 50]); // from generation_config.json's eos_token_id list
  for (let step = 0; step < maxNewTokens; step++) {
    const output = await session.run(feeds);
    const logitsTensor = output.logits; // [1, 1, vocab]
    // logitsTensor.data is a native Float16Array — already real decoded float values, not raw fp16
    // bit patterns (verified: values like -12.43/13.42 are sane transformer logit magnitudes).
    const logits = logitsTensor.data;
    let bestIdx = 0, bestVal = -Infinity;
    for (let v = 0; v < logits.length; v++) {
      if (logits[v] > bestVal) { bestVal = logits[v]; bestIdx = v; }
    }
    if (eosIds.has(bestIdx)) break;
    generated.push(bestIdx);
    allTokenIds.push(bestIdx);

    const nextPastKV = {};
    for (let i = 0; i < NUM_DECODER_LAYERS; i++) {
      nextPastKV[`past_key_values.${i}.key`] = output[`present.${i}.key`];
      nextPastKV[`past_key_values.${i}.value`] = output[`present.${i}.value`];
    }

    const nextEmb = await embedTokensSparse([bestIdx]);
    seqLen = allTokenIds.length;
    feeds = {
      inputs_embeds: new ort.Tensor("float32", nextEmb.inputsEmbeds, [1, 1, DIMS.MAIN_DIM]),
      per_layer_inputs: new ort.Tensor("float32", nextEmb.perLayerInputs, [1, 1, DIMS.NUM_LAYERS, DIMS.PLE_LAYER_DIM]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(seqLen).fill(1n)), [1, seqLen]),
      position_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(seqLen - 1)]), [1, 1]),
      num_logits_to_keep: new ort.Tensor("int64", BigInt64Array.from([1n]), []),
      ...nextPastKV,
    };

    // stream partial decode to the UI
    const partial = generated.length ? tokenizer.decode(generated, { skip_special_tokens: true }) : "";
    onToken(partial);
  }
  return generated.length ? tokenizer.decode(generated, { skip_special_tokens: true }) : "(model produced no tokens — see console for top5 logits)";
}

// Slice a KV-cache tensor along the sequence axis, keeping only the first `keepLen` positions.
// Safe because this model uses multi-query attention (1 KV head) and batch=1, so a tensor of
// dims [1,1,seqLen,dim] is just `seqLen` contiguous blocks of `dim` values — a plain typed-array
// slice, no reshaping/striding math needed.
function sliceKV(tensor, keepLen, dim) {
  const data = tensor.data.slice(0, keepLen * dim);
  return new ort.Tensor(tensor.type, data, [1, 1, keepLen, dim]);
}

function argmaxAt(logitsTensor, position, vocabSize) {
  const offset = position * vocabSize;
  const data = logitsTensor.data;
  let bestIdx = 0, bestVal = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    if (data[offset + v] > bestVal) { bestVal = data[offset + v]; bestIdx = v; }
  }
  return bestIdx;
}

// Speculative decoding: a small draft model (Gemma 3 270M) proposes several tokens ahead; the
// target model (Gemma 4 E2B) verifies all of them in ONE forward pass instead of one-per-token.
// Verification is exact greedy-argmax equality (not the full stochastic rejection-sampling
// scheme from Leviathan et al. — that variant is for matching a SAMPLING distribution; ours is
// plain greedy decoding, so exact-match verification is both correct and simpler: it provably
// produces the identical token sequence plain greedy decoding would, just potentially faster).
const SPEC_K = 4; // 1 known-good "seed" token (priorGuess) + up to 3 genuinely speculative ones

async function generateSpeculative(promptTokenIds, maxNewTokens = 220) {
  const eosIds = new Set([1, 106, 50]);
  let allTokenIds = [...promptTokenIds];
  let generated = [];

  // --- initial prompt forward pass (same as plain generate()) — gives us pastKV + priorGuess ---
  let { inputsEmbeds, perLayerInputs } = await embedTokensSparse(allTokenIds);
  let seqLen = allTokenIds.length;
  let output = await session.run({
    inputs_embeds: new ort.Tensor("float32", inputsEmbeds, [1, seqLen, DIMS.MAIN_DIM]),
    per_layer_inputs: new ort.Tensor("float32", perLayerInputs, [1, seqLen, DIMS.NUM_LAYERS, DIMS.PLE_LAYER_DIM]),
    attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(seqLen).fill(1n)), [1, seqLen]),
    position_ids: new ort.Tensor("int64", BigInt64Array.from(Array.from({ length: seqLen }, (_, i) => BigInt(i))), [1, seqLen]),
    num_logits_to_keep: new ort.Tensor("int64", BigInt64Array.from([1n]), []),
    ...emptyKV(),
  });
  let pastKV = {};
  for (let i = 0; i < NUM_DECODER_LAYERS; i++) {
    pastKV[`past_key_values.${i}.key`] = output[`present.${i}.key`];
    pastKV[`past_key_values.${i}.value`] = output[`present.${i}.value`];
  }
  let priorGuess = argmaxAt(output.logits, 0, output.logits.dims[2]);

  while (generated.length < maxNewTokens) {
    if (eosIds.has(priorGuess)) break;

    const speculative = await draftTokens([...allTokenIds, priorGuess], SPEC_K - 1, eosIds);
    const batchTokens = [priorGuess, ...speculative]; // length 1..SPEC_K
    const batchLen = batchTokens.length;

    const emb = await embedTokensSparse(batchTokens);
    const verifyOut = await session.run({
      inputs_embeds: new ort.Tensor("float32", emb.inputsEmbeds, [1, batchLen, DIMS.MAIN_DIM]),
      per_layer_inputs: new ort.Tensor("float32", emb.perLayerInputs, [1, batchLen, DIMS.NUM_LAYERS, DIMS.PLE_LAYER_DIM]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(seqLen + batchLen).fill(1n)), [1, seqLen + batchLen]),
      position_ids: new ort.Tensor("int64", BigInt64Array.from(Array.from({ length: batchLen }, (_, i) => BigInt(seqLen + i))), [1, batchLen]),
      num_logits_to_keep: new ort.Tensor("int64", BigInt64Array.from([BigInt(batchLen)]), []),
      ...pastKV,
    });
    const vocabSize = verifyOut.logits.dims[2];

    let firstFailAt = null; // index into [0, batchLen-2] of the first wrong speculative token
    for (let i = 0; i < batchLen - 1; i++) {
      if (argmaxAt(verifyOut.logits, i, vocabSize) !== batchTokens[i + 1]) { firstFailAt = i; break; }
    }

    let acceptedCount, kvKeepLen;
    if (firstFailAt === null) {
      // every speculative token was right — accept all of them, plus a free bonus prediction
      acceptedCount = batchLen;
      kvKeepLen = seqLen + batchLen;
      generated.push(...batchTokens);
      allTokenIds.push(...batchTokens);
      priorGuess = argmaxAt(verifyOut.logits, batchLen - 1, vocabSize);
      pastKV = {};
      for (let i = 0; i < NUM_DECODER_LAYERS; i++) {
        pastKV[`past_key_values.${i}.key`] = verifyOut[`present.${i}.key`];
        pastKV[`past_key_values.${i}.value`] = verifyOut[`present.${i}.value`];
      }
    } else {
      // accept the confirmed prefix, replace the wrong token with the target's own correction
      const accepted = batchTokens.slice(0, firstFailAt + 1);
      const corrected = argmaxAt(verifyOut.logits, firstFailAt, vocabSize);
      generated.push(...accepted, corrected);
      allTokenIds.push(...accepted, corrected);
      kvKeepLen = seqLen + firstFailAt + 1; // trim off the rejected tail
      const trimmedKV = {};
      for (let i = 0; i < NUM_DECODER_LAYERS; i++) {
        trimmedKV[`past_key_values.${i}.key`] = sliceKV(verifyOut[`present.${i}.key`], kvKeepLen, KV_DIM(i));
        trimmedKV[`past_key_values.${i}.value`] = sliceKV(verifyOut[`present.${i}.value`], kvKeepLen, KV_DIM(i));
      }
      // one normal single-token step for `corrected`, to get both a fresh priorGuess and a KV
      // cache that actually reflects it (the batch above didn't compute anything meaningful past
      // the rejection point, since it used the WRONG token as context beyond firstFailAt).
      const correctedEmb = await embedTokensSparse([corrected]);
      const correctionOut = await session.run({
        inputs_embeds: new ort.Tensor("float32", correctedEmb.inputsEmbeds, [1, 1, DIMS.MAIN_DIM]),
        per_layer_inputs: new ort.Tensor("float32", correctedEmb.perLayerInputs, [1, 1, DIMS.NUM_LAYERS, DIMS.PLE_LAYER_DIM]),
        attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array(kvKeepLen + 1).fill(1n)), [1, kvKeepLen + 1]),
        position_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(kvKeepLen)]), [1, 1]),
        num_logits_to_keep: new ort.Tensor("int64", BigInt64Array.from([1n]), []),
        ...trimmedKV,
      });
      priorGuess = argmaxAt(correctionOut.logits, 0, correctionOut.logits.dims[2]);
      pastKV = {};
      for (let i = 0; i < NUM_DECODER_LAYERS; i++) {
        pastKV[`past_key_values.${i}.key`] = correctionOut[`present.${i}.key`];
        pastKV[`past_key_values.${i}.value`] = correctionOut[`present.${i}.value`];
      }
      kvKeepLen += 1;
    }
    seqLen = kvKeepLen;

    const partial = generated.length ? tokenizer.decode(generated, { skip_special_tokens: true }) : "";
    onToken(partial);
    if (generated.some((t) => eosIds.has(t))) break;
  }

  // strip any trailing EOS-family tokens that slipped into `generated` before the loop noticed
  const clean = generated.filter((t) => !eosIds.has(t));
  return clean.length ? tokenizer.decode(clean, { skip_special_tokens: true }) : "(model produced no tokens)";
}

let onToken = () => {};

async function send() {
  const text = inputEl.value.trim();
  if (!text || !session) return;
  inputEl.value = "";
  sendBtn.disabled = true;
  addMsg("user", text);
  history.push({ role: "user", content: text });

  const botDiv = addMsg("bot", "");
  onToken = (partial) => {
    botDiv.textContent = partial;
    chatEl.scrollTop = chatEl.scrollHeight;
  };

  try {
    const prompt = buildChatPrompt(history);
    const { input_ids } = await tokenizer(prompt, { return_tensor: false });
    const rawIds = Array.isArray(input_ids[0]) ? input_ids[0] : input_ids;
    const BOS_ID = 2; // tokenizer_config.json: add_bos_token is unset, so raw tokenizer() calls
    // don't prepend <bos> automatically — Gemma models expect every sequence to start with it.
    const promptTokenIds = rawIds[0] === BOS_ID ? rawIds : [BOS_ID, ...rawIds];
    const full = draftReady ? await generateSpeculative(promptTokenIds) : await generate(promptTokenIds);
    history.push({ role: "assistant", content: full });
  } catch (err) {
    botDiv.textContent = `(generation error: ${err?.message || err})`;
    console.error(err);
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

init();
