# The embedding table is 31–56% of every popular browser LLM download — and almost none of it is used

This is the result that makes SparseGemma's technique matter at ecosystem scale rather than for one
model. **Measured directly** by inspecting the ONNX graphs of the most-downloaded browser-deployable
small LLMs with `onnx` + byte-offset verification (scripts in `survey/`).

## The finding

| Model | Total download | Embedding table | **Embedding = % of download** | Storage | Input range-fetch | Output head |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** (q4f16) | 483 MB | 272 MB (**fp16, un-quantized**) | **56.4%** | inline, single `.onnx` | yes (offset proven) | tied |
| **SmolLM2-360M-Instruct** (q4f16) | 273 MB | 94 MB (fp16) | **34.6%** | inline, single `.onnx` | yes | tied |
| **Gemma-3-270M-it** (q4f16) | 273 MB | 84 MB (quantized) | **30.8%** | external `.onnx_data` | yes (direct) | tied |
| **Gemma-4-E2B-it** (q4f16) | ~3.1 GB | 1.59 GB (quant + PLE) | ~51% | external `.onnx_data` | yes (**live demo**) | separate/internal |

A conversation touches only a few hundred of each model's tens-to-hundreds of thousands of vocabulary
rows (Heaps' law, β≈0.74; see `THEORY.md`). Concretely, for Qwen2.5-0.5B: **400 unique tokens = 700 KB
vs the 272 MB full table = 0.263%.** Across these four of the most popular browser LLMs, **31–56% of every
download is an embedding table where <0.3% of the rows are ever read.**

The most striking case: **Qwen2.5-0.5B ships more than half its total download as a *fully un-quantized
fp16* embedding table.** That is two inefficiencies stacked — huge *and* not even compressed.

## The tied-weights subtlety (this is where most "just fetch less" claims break — and we checked)

**Critical caveat, verified from the graphs, not assumed.** In all three single-file models above the
embedding table is **weight-tied to the output head**: `embed_tokens.weight → Transpose → /lm_head/MatMul
→ logits` (Qwen, SmolLM2), and Gemma-3-270M's quantized embedding feeds both `GatherBlockQuantized`
(input) and `MatMulNBits` (output). So the table is used **twice**: once as an input lookup (one row per
input token) and once as the output projection (a score for *every* vocab row).

Consequences — stated precisely, because this is exactly what a framework maintainer will ask:

- **Input embedding lookup — sparse fetch applies universally.** Only the rows for input tokens are read;
  this is the Heaps-law sublinear win, for every model surveyed.
- **Output logits projection — depends on generation strategy and export:**
  - *Separate / internal LM head* (e.g. Gemma-4-E2B's `decoder_model_merged` export, which computes
    logits inside the decoder file — this is **why the SparseGemma demo works end-to-end today**): the
    embedding file is input-only; sparse fetch is fully sufficient.
  - *Tied head + full-vocabulary argmax*: the output MatMul needs **all** rows → the full table is
    required for output. Sparse fetch still saves the input side but cannot avoid the table here.
  - *Tied head + candidate-set generation* (speculative decoding / draft-guided / top-k): logits are only
    needed for the **candidate tokens the draft proposes**, i.e. a handful of rows — so the output
    projection becomes sparse too, over the *same* rows sparse fetch already handles. **This is Theorem 2
    doing triple duty**: the draft model is simultaneously the latency mechanism (spec decoding), the
    *input*-embedding prefetch oracle, and the selector of which *output*-head rows are needed.

Bottom line: the *"31–56% of the download is the embedding table"* measurement stands unconditionally;
whether you can *skip* downloading it is clean for separate-head exports and for draft-guided generation,
and is a genuine limitation for exhaustive full-vocab argmax on a tied single-file export. We do not
paper over this.

## Two storage layouts — the technique covers both (input side; output side per the caveat above)

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
  contiguity (byte-for-byte checked); that all three single-file models have **tied output heads**
  (traced `embed → Transpose → /lm_head/MatMul → logits`, `survey/trace_output.py`).
- **Demonstrated end-to-end:** the external-data, separate-head path (SparseGemma live on Gemma-4-E2B).
- **Proven-feasible, not yet wired into the demo:** the inline-file input path (offset located & verified;
  a browser fetch of that sub-range + fp16 reshape is remaining engineering, not a research risk).
- **Genuine limitation, stated:** for a *tied-head* export under *full-vocabulary argmax*, the output
  projection needs the whole table; the sparse-output win requires candidate-set (draft-guided/top-k)
  generation. Not hidden — it's the central caveat above.
- **Not claimed:** that every ONNX LLM everywhere uses one of these layouts — four popular models are
  surveyed; the method to check a new one is `survey/survey.py` + `survey/trace_output.py` (reproducible).

## Reproduce
```bash
pip install onnx
python3 survey/survey.py        # embedding tensor + external-data fraction per model
python3 survey/measure_frac.py  # inline-embedding fraction (Qwen, SmolLM2)
python3 survey/prove_offset.py  # prove the inline embedding is contiguous & range-fetchable
```
