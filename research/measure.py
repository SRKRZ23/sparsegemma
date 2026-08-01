#!/usr/bin/env python3
"""Empirically measure token-access statistics that govern SparseGemma's embedding-fetch cost.

Measures, on a real diverse conversation corpus tokenized with the exact Gemma tokenizer:
  (1) HEAPS' LAW  — unique tokens U(n) ~ K n^β  → fetch traffic is SUBLINEAR in conversation length.
  (2) ZIPF-MANDELBROT — rank-frequency exponent s → governs cache reuse.
  (3) LRU CACHE HIT RATE — measured, vs the analytical Che-approximation prediction from s.
       Agreement means classical caching theory *predicts* our system's real behavior.

Everything here is measured, not assumed. Numbers are written to results.json for the writeup.
"""
import json, math, pathlib
from collections import Counter
import numpy as np
from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).parent
tok = Tokenizer.from_file(str(HERE / "tokenizer.json"))
# Read every available corpus (Wikipedia real-human-text + any LLM-generated chat).
files = sorted(HERE.glob("corpus*/*.txt"))
files = [f for f in files if f.name != "manifest.json"]
assert files, "no corpus yet"

# --- tokenize every conversation into an ordered token-id stream ---
streams = []
for f in files:
    ids = tok.encode(f.read_text()).ids
    if len(ids) >= 50:
        streams.append(ids)
print(f"{len(streams)} conversations, "
      f"{sum(len(s) for s in streams)} total tokens, "
      f"median length {int(np.median([len(s) for s in streams]))}")

# ---------- (1) HEAPS' LAW: U(n) ~ K n^β ----------
# Fit per-conversation in log-log space, and also on the pooled (concatenated) stream.
def heaps_beta(stream):
    seen = set(); U = []
    for t in stream:
        seen.add(t); U.append(len(seen))
    n = np.arange(1, len(U) + 1)
    U = np.array(U)
    # fit for n >= 10 (Heaps' law is asymptotic; tiny-n is dominated by the trivial U≈n regime)
    m = n >= 10
    if m.sum() < 10:
        return None
    logn, logU = np.log(n[m]), np.log(U[m])
    beta, logK = np.polyfit(logn, logU, 1)
    # R^2
    pred = logK + beta * logn
    ss_res = np.sum((logU - pred) ** 2); ss_tot = np.sum((logU - logU.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta, math.exp(logK), r2

betas = [heaps_beta(s) for s in streams]
betas = [b for b in betas if b is not None]
beta_vals = np.array([b[0] for b in betas])
r2_vals = np.array([b[2] for b in betas])

# pooled stream (one long "session")
pooled = [t for s in streams for t in s]
pb = heaps_beta(pooled)

print("\n=== HEAPS' LAW  U(n) ~ K n^beta ===")
print(f"per-conversation beta: mean={beta_vals.mean():.4f}  std={beta_vals.std():.4f}  "
      f"median={np.median(beta_vals):.4f}  (mean R^2={r2_vals.mean():.4f})")
print(f"pooled-stream beta={pb[0]:.4f}  K={pb[1]:.2f}  R^2={pb[2]:.4f}")

# ---------- (2) ZIPF-MANDELBROT rank-frequency ----------
freq = Counter(pooled)
ranked = np.array(sorted(freq.values(), reverse=True), dtype=float)
ranks = np.arange(1, len(ranked) + 1)
# fit f ~ C / rank^s on the body (ranks 10..min(len,20000)) in log-log
lo, hi = 10, min(len(ranked), 20000)
lr, lf = np.log(ranks[lo:hi]), np.log(ranked[lo:hi])
neg_s, logC = np.polyfit(lr, lf, 1)
s_zipf = -neg_s
print("\n=== ZIPF-MANDELBROT  f ~ C / rank^s ===")
print(f"exponent s={s_zipf:.4f}  (unique tokens overall={len(ranked)} of 262144 vocab = "
      f"{100*len(ranked)/262144:.2f}%)")

# ---------- (3) LRU hit rate: measured vs Che-approximation from s ----------
# Che's approximation: for an LRU cache of size C over an IRM (independent reference model) with
# per-item request probabilities p_i, define the "characteristic time" t_C as the unique root of
#   sum_i (1 - exp(-p_i t_C)) = C ,
# then item i's hit probability ≈ 1 - exp(-p_i t_C), and overall hit rate = sum_i p_i (1-exp(-p_i t_C)).
probs = ranked / ranked.sum()
def che_hit_rate(C):
    # solve sum_i 1 - exp(-p_i t) = C for t by bisection
    lo_t, hi_t = 0.0, 1e9
    for _ in range(100):
        t = 0.5 * (lo_t + hi_t)
        occ = np.sum(1 - np.exp(-probs * t))
        if occ < C: lo_t = t
        else: hi_t = t
    t = 0.5 * (lo_t + hi_t)
    return float(np.sum(probs * (1 - np.exp(-probs * t))))

def measured_lru_hit_rate(stream, C):
    # exact LRU over the real ordered stream (not IRM) — the ground truth Che approximates
    from collections import OrderedDict
    cache = OrderedDict(); hits = 0; total = 0
    for t in stream:
        total += 1
        if t in cache:
            hits += 1; cache.move_to_end(t)
        else:
            cache[t] = True
            if len(cache) > C: cache.popitem(last=False)
    return hits / total

# RIGOR CONTROL: shuffle the pooled stream to destroy temporal locality while preserving the exact
# token frequencies — this yields a true independent-reference (IRM) stream. If Che matches the
# SHUFFLED stream but the REAL ordered stream beats it, the gap is genuine temporal locality
# (burstiness/topic locality), not a bug in the Che implementation. This is the decisive check.
rng = np.random.default_rng(0)
shuffled = pooled.copy()
rng.shuffle(shuffled)

print("\n=== LRU CACHE HIT RATE: real ordered stream vs IRM-shuffled vs Che-approx ===")
print("  (Che should match the IRM-shuffled stream; real>IRM ⇒ genuine temporal locality)")
lru_rows = []
for C in [64, 128, 256, 512, 1024, 2048]:
    measured = measured_lru_hit_rate(pooled, C)
    shuf = measured_lru_hit_rate(list(shuffled), C)
    che = che_hit_rate(C)
    print(f"  cache={C:5d}: real={measured:.4f}  irm_shuffled={shuf:.4f}  che_approx={che:.4f}  "
          f"|che-irm|={abs(che-shuf):.4f}  locality_gain(real-irm)={measured-shuf:+.4f}")
    lru_rows.append({"cache": C, "measured_real": measured, "irm_shuffled": shuf, "che_approx": che})

results = {
    "n_conversations": len(streams),
    "total_tokens": len(pooled),
    "heaps_beta_mean": float(beta_vals.mean()),
    "heaps_beta_std": float(beta_vals.std()),
    "heaps_beta_pooled": float(pb[0]),
    "heaps_K_pooled": float(pb[1]),
    "heaps_r2_mean": float(r2_vals.mean()),
    "zipf_s": float(s_zipf),
    "unique_tokens_total": int(len(ranked)),
    "vocab": 262144,
    "lru": lru_rows,
}
(HERE / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {HERE/'results.json'}")
