# RFC: On-demand (token-sparse) embedding loading for browser LLMs

**Target projects:** Transformers.js (`@huggingface/transformers`) primarily; onnxruntime-web as the
lower-level enabler.
**Status:** proposal + working reference implementation (SparseGemma) + reproducible measurements.
**Author:** external contribution. Everything below is measured; scripts to reproduce are linked.

---

## 1. The problem, measured

For the most-downloaded browser-deployable LLMs, a large fraction of the total model download is the
token-embedding table — and a chat session reads a tiny fraction of its rows.

| Model (q4f16 ONNX) | Total | Embedding | **% of download** |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 483 MB | 272 MB (fp16, un-quantized) | **56.4%** |
| SmolLM2-360M-Instruct | 273 MB | 94 MB | 34.6% |
| Gemma-3-270M-it | 273 MB | 84 MB | 30.8% |
| Gemma-4-E2B-it | ~3.1 GB | 1.59 GB | ~51% |

Token access follows **Heaps' law** (measured β≈0.74, R²=0.999 on a 164k-token corpus with the exact
Gemma tokenizer): a 1000-token conversation touches ≈412 distinct tokens. For Qwen2.5-0.5B that is
**700 KB of embedding rows vs a 272 MB table = 0.26%.** Today every browser user downloads the whole
table before the first token. Reproduce: `research/survey/*.py`, `research/measure.py`.

## 2. Proposed mechanism

The embedding table in these ONNX exports is a contiguous, row-major tensor: token id → fixed byte
range. Instead of downloading it whole, **fetch each token's row on demand via an HTTP `Range` request,
and cache it** (in-memory + IndexedDB for cross-session reuse). HTTP Range is universally supported by the
HF CDN (`Accept-Ranges: bytes`, verified). Two layouts, both handled:

- **External-data blob** (`model.onnx_data`): row `r` = `Range: bytes = base + r·rowBytes … +rowBytes-1`.
  Offsets come straight from the initializer's `external_data{offset,length}`.
- **Inline single-file** (`model.onnx`): the tensor's `raw_data` is contiguous inside the protobuf; parse
  the graph once to get its file offset, then Range-fetch the sub-range. Verified byte-for-byte on
  Qwen2.5-0.5B (`research/survey/prove_offset.py`).

Dequantization (for quantized tables) is the op's documented formula — for Gemma's
`GatherBlockQuantized`: 4-bit nibble-packed, `block_size` groups, per-group scale + zero-point, global
scale. Verified byte-identical to full dequantization.

## 3. The one subtlety that matters — tied output heads (addressed, not hidden)

In the single-file models above the embedding is **weight-tied to the LM head**
(`embed → Transpose → /lm_head/MatMul → logits`; traced in `research/survey/trace_output.py`). So the
table is used for input lookup *and* output logits.

- **Input side:** on-demand fetch always applies.
- **Output side:** full-vocab argmax needs the whole table for the final MatMul. Two clean resolutions:
  1. **Separate/internal LM-head exports** (e.g. Gemma-4-E2B's `decoder_model_merged`, which emits logits
     internally) — input embedding is fetch-only; this is what the reference demo runs end-to-end today.
  2. **Candidate-set generation** (speculative decoding / top-k): logits are needed only for the few
     candidate tokens, i.e. only those output-head rows — the same rows already fetched. A draft model
     both proposes candidates and (Theorem 2, `research/THEORY.md`) bounds the input-prefetch miss rate.

Recommended integration path: expose on-demand embedding fetch as an **opt-in loader flag**, correct by
construction for separate-head exports and for candidate-set decoding; fall back to full download for
tied-head + full-vocab-argmax. This is a strict superset of current behavior — never worse.

## 4. API sketch (Transformers.js)

```js
const model = await AutoModelForCausalLM.from_pretrained(id, {
  dtype: "q4f16",
  device: "webgpu",
  embeddings: "on-demand",          // NEW: range-fetch rows lazily instead of downloading the table
  embeddingCache: "indexeddb",      // NEW: persist fetched rows across sessions
});
```
Under the hood: a small `SparseEmbedding` provider that (a) reads the embedding initializer's
offset/length or inline file-offset at load, (b) intercepts the embedding-Gather with a cached
Range-fetch, (c) leaves the rest of the graph untouched. onnxruntime-web already supports partial/external
data; the lower-level enabler is a documented hook to supply an embedding tensor lazily rather than
materializing it up front.

## 5. Impact

- **Download:** −30% to −56% of model bytes for the surveyed models (input-only side; full for
  separate-head/candidate-set decoding).
- **Time to first token on slow links:** dominated today by the embedding download; on-demand fetch makes
  it proportional to prompt size, not vocab size.
- **Aggregate:** at ecosystem scale (millions of browser-LLM sessions), each avoiding 100–270 MB is
  petabyte-scale CDN traffic, real cost, and real energy — see `research/IMPACT.md` for the arithmetic.
- **Compute:** zero extra serve-time compute; caching makes repeat tokens free across sessions.

## 6. Limitations (complete)

- Many small Range requests vs one bulk download: mitigated by a worker-pool + prefetch; on very fast
  links a bulk download of a small table can still be faster in wall-clock — hence *opt-in*.
- Tied-head + full-vocab argmax: needs the full table for output (§3).
- Un-quantized fp16 inline tables (Qwen, SmolLM2) are separately wasteful — quantizing them is orthogonal
  and complementary.

## 7. Reference implementation & reproduction
- Working browser demo (Gemma-4-E2B, 1.59 GB table → hundreds of KB fetched): https://sparsegemma.netlify.app
- Code: https://github.com/SRKRZ23/sparsegemma
- Survey / measurements / proofs: `research/` (all scripts reproducible with `pip install onnx tokenizers numpy`).
