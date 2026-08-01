# SparseGemma — token-sparse embedding loading for on-device LLMs

A browser-based proof that a 2.9GB+ model's embedding table doesn't have to be downloaded in full to run
on-device. **SparseGemma** fetches only the byte ranges for tokens actually used in a conversation —
verified to cut the embedding download from **1.59GB to a few KB** for a typical prompt, with zero quality
loss (byte-identical output to full dequantization, verified against the model's own weights).

## The invention

Gemma 4's E2B checkpoint (`onnx-community/gemma-4-E2B-it-ONNX`) stores its embedding table as a plain
row-major lookup: token id → fixed byte offset. Every serving stack that loads this model — llama.cpp,
Google's own LiteRT-LM, ONNX Runtime GenAI, Transformers.js — downloads and/or memory-maps the **entire**
table before inference, because none of them exploit the fact that a real conversation only ever touches a
few hundred of the 262,144 possible tokens.

SparseGemma instead:
1. Reverse-engineers the exact `GatherBlockQuantized` dequantization math (4-bit nibble-packed weights,
   block_size=32 groups, per-group scale + zero-point, a final global scale multiply) directly from the
   ONNX graph — verified byte-for-byte against a Python reference implementation.
2. HTTP-Range-fetches only the rows for tokens present in the current input, for both the main embedding
   table and Gemma's Per-Layer-Embedding (PLE) table, in parallel.
3. Feeds the resulting embeddings straight into a custom, hand-rolled decoder generation loop (raw
   `onnxruntime-web` session, manual KV-cache management) — bypassing the high-level
   `Gemma4ForConditionalGeneration` API entirely, since that API always loads the full embedding table
   regardless of what you actually need.
4. Caches every fetched token's embedding in IndexedDB, so repeat tokens — across the *whole browser
   lifetime*, not just one page load — never cost a network round-trip again.

Net effect on a real prompt (measured, not estimated): **835KB fetched for a 231-token prompt on first
use; 0KB on every use after** (IndexedDB warm). The dense decoder weights (~1.5GB) still have to be
downloaded in full — there's no equivalent sparsity trick for those (no MoE in this architecture, just
PLE) — but the embedding table, previously the single largest file in the stack, is no longer a fixed cost.

## Demo application: PocketTriage

The showcase app is a private, on-device health-navigation companion — decision-support and education
only (explains general urgency language the way ED triage systems reason about it, suggests questions for
a real clinician, never diagnoses). Chosen because it's exactly the kind of use case where "nothing you
type ever leaves the device" is a real requirement, not a nice-to-have.

## Verified, not claimed

```
[sparse_embed] 141 unique tokens, 835.5KB fetched (vs 1.59GB full table)
...
step 0: top5 (id,val) = [[106,13.89],[1,13.42],[3048,12.98],...]   ← real, peaked transformer logits
...
"Urgency Category: Soon. Why: Mild headaches that last for a couple of days without other
concerning symptoms are often non-urgent... Questions for a Clinician: ... Disclaimer: ..."
```
Full end-to-end generation, real WebGPU execution, coherent on-topic output — captured via automated
browser testing (Playwright + headless Chrome with WebGPU enabled), not hand-waved.

## Architecture
```
index.html + app.js + sparse_embed.js  (vanilla JS, ES modules, no build step)
        │
        ├── sparse_embed.js ── HTTP Range fetch + LUT-based dequant (T-MAC-style) for only the
        │                       tokens actually used → inputs_embeds + per_layer_inputs tensors
        │                       (IndexedDB-cached across sessions)
        │
        └── onnxruntime-web (raw, not via Transformers.js's model class)
                │
                ▼
        decoder_model_merged_q4f16.onnx (~1.5GB, dense — no sparsity trick applies here)
                │
                ▼
        WebGPU ── all decode-loop compute happens here, on-device
```

## Honest status
- Core pipeline verified working end-to-end (see captured output above).
- The dense decoder weights are still a real ~1.5GB one-time download — this is a hard floor for
  post-training-quantized dense models at this quality tier, not something this technique addresses.
- Prewarming common vocabulary at idle time + a priority-aware fetch scheduler (background prewarm yields
  to an active user request) are both implemented and tested.
- No image/audio modality wired up (the underlying model supports it) — text chat only, kept in scope.

## Run locally
```bash
cd public && python3 -m http.server 8834
# open http://localhost:8834 in a WebGPU-capable browser (Chrome/Edge, recent version)
```

## Live demo
https://sparsegemma.netlify.app (WebGPU browser required)

## License
MIT. Take the sparse-loading technique, use it anywhere.
