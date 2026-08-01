# SparseGemma — token-access statistics & the draft-oracle prefetch bound

Empirical + theoretical companion to SparseGemma. **Everything here is measured on a real corpus with
the exact Gemma tokenizer; every claim states its assumptions.** See `THEORY.md` for the full write-up
with proofs and the honesty ledger.

## Headline results (measured, real corpus, 164,281 tokens, exact Gemma tokenizer)

| Result | Value | Meaning for SparseGemma |
|---|---|---|
| **Heaps' law** β | **0.741** (R²=0.999) | Embedding traffic is *sublinear* in conversation length — a 1000-token chat touches ~412 distinct tokens ⇒ **2.5 MB fetched, 0.16% of the 1.59 GB table** |
| **Zipf** s | **1.086** | Textbook heavy tail; only **6.6%** of the 262k vocab appears at all |
| **Che ≈ IRM** | within **0.012** | Classical closed-form cache model validated against its own assumptions (shuffle control) |
| **Locality gain** | **+0.15** hit rate | Real streams beat the frequency-only IRM prediction — genuine temporal locality LRU exploits; **a 256-row (~1.5 MB) cache already hits 61%** |
| **Draft acceptance** β_acc | **≈0.40** (preliminary) | Speculative-decoding acceptance rate *equals* the top-1 prefetch cold-miss complement (Theorem 2) — the draft model is a free prefetch oracle |

## The one genuinely new idea (Theorem 2)
The small draft model we already run for speculative decoding produces, at zero extra compute, a next-
token distribution. Prefetching its top-k gives embedding-prefetch coverage, and **for greedy decoding the
top-1 prefetch cold-miss rate is *exactly* one minus the speculative-decoding acceptance rate.** One
measured scalar ties the latency technique and the bandwidth technique together. Not previously stated.

## Figures
- `figures/fig1_heaps.png` — U(n) ∝ n^0.74 across 5 orders of magnitude (R²=0.999)
- `figures/fig2_zipf.png` — rank-frequency heavy tail (s≈1.09)
- `figures/fig3_lru_locality.png` — Che validated vs IRM; real stream's +0.15 locality margin

## Reproduce
```bash
pip install tokenizers numpy matplotlib scipy
python3 fetch_wikipedia.py      # real diverse human-text corpus (public API)
python3 measure.py              # → results.json (Heaps β, Zipf s, LRU real/IRM/Che)
python3 plots.py                # → figures/*.png
```
`generate_corpus.py` additionally builds an LLM-generated multi-turn *chat* corpus (rate-limited free
API) for a chat-domain cross-check of the exponents; prose is the harder, richer-vocabulary case so its
β likely upper-bounds chat's.

## Files
- `THEORY.md` — theorems, proofs, classical-vs-novel separation, honesty ledger
- `measure.py` — Heaps/Zipf fits + LRU real/IRM-shuffle/Che validation
- `plots.py` — the three figures
- `results.json` — raw measured numbers
