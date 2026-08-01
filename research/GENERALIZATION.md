# The embedding table is 31–56% of every popular browser LLM download — and almost none of it is used

This is the result that makes SparseGemma's technique matter at ecosystem scale rather than for one
model. **Measured directly** by inspecting the ONNX graphs of the most-downloaded browser-deployable
small LLMs with `onnx` + byte-offset verification (scripts in `survey/`).

## The finding

| Model | Total download | Embedding table | **Embedding = % of download** | Storage | Range-fetchable |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** (q4f16) | 483 MB | 272 MB (**fp16, un-quantized**) | **56.4%** | inline, single `.onnx` | yes (proven, see below) |
| **SmolLM2-360M-Instruct** (q4f16) | 273 MB | 94 MB (fp16) | **34.6%** | inline, single `.onnx` | yes |
| **Gemma-3-270M-it** (q4f16) | 273 MB | 84 MB (quantized) | **30.8%** | external `.onnx_data` | yes (direct) |
| **Gemma-4-E2B-it** (q4f16) | ~3.1 GB | 1.59 GB (quant + PLE) | ~51% | external `.onnx_data` | yes (the SparseGemma demo) |

A conversation touches only a few hundred of each model's tens-to-hundreds of thousands of vocabulary
rows (Heaps' law, β≈0.74; see `THEORY.md`). Concretely, for Qwen2.5-0.5B: **400 unique tokens = 700 KB
vs the 272 MB full table = 0.263%.** Across these four of the most popular browser LLMs, **31–56% of every
download is an embedding table where <0.3% of the rows are ever read.**

The most striking case: **Qwen2.5-0.5B ships more than half its total download as a *fully un-quantized
fp16* embedding table.** That is two inefficiencies stacked — huge *and* not even compressed.

## Two storage layouts — the technique covers both

**(A) External-data quantized** (Gemma family). The `.onnx` graph is tiny (KB); weights live in a
separate `.onnx_data` blob; each token's embedding row is a fixed byte range in that blob. Direct HTTP
Range fetch — this is exactly what the SparseGemma demo does.

**(B) Inline single-file** (Qwen, SmolLM2). The whole model, embedding included, is one `.onnx` protobuf
file with no separate data blob. Less obviously range-fetchable — **but proven so.** The embedding
tensor's `raw_data` is stored *contiguously* inside the protobuf; parsing the graph once yields its exact
byte offset, after which any row is a Range fetch of the single file:

```
Qwen2.5-0.5B: embedding raw_data occupies bytes [548477, 272817789) of model_q4f16.onnx
  — VERIFIED contiguous & byte-for-byte matching.
  Row r of token embedding = bytes [548477 + r*1792, +1791]   (fp16, 896 dims/row).
```
(`survey/prove_offset.py` locates the offset by a unique mid-tensor byte signature and verifies the full
range matches the loaded tensor exactly.)

## Why this is the "can't-ignore" version

- It is not one model. It is a **systematic inefficiency across the most-downloaded browser LLMs**, each
  independently measured.
- The fix is **already built and demonstrated end-to-end** (SparseGemma, on the hardest case — Gemma-4-
  E2B's 1.59 GB table) and **provably generalizes** to both ONNX storage layouts and to un-quantized
  inline embeddings.
- It is **compute-free at serve time** and, combined with the draft-oracle prefetch (Theorem 2,
  `THEORY.md`), turns per-token embedding latency from network-bound to hidden.

## Honesty ledger
- **Measured, verified:** the four models' embedding fractions; Qwen's exact inline byte offset and
  contiguity (byte-for-byte checked).
- **Demonstrated end-to-end:** the external-data path (SparseGemma live on Gemma-4-E2B).
- **Proven-feasible but not yet wired into the demo:** the inline-file path (offset located & verified;
  a browser fetch of that sub-range + fp16 reshape is the remaining engineering, not a research risk).
- **Not claimed:** that every ONNX LLM everywhere uses one of these two layouts — four popular models are
  surveyed; the method to check a new one is `survey/survey.py` (reproducible).

## Reproduce
```bash
pip install onnx
python3 survey/survey.py        # embedding tensor + external-data fraction per model
python3 survey/measure_frac.py  # inline-embedding fraction (Qwen, SmolLM2)
python3 survey/prove_offset.py  # prove the inline embedding is contiguous & range-fetchable
```
