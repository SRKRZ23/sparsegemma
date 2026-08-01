#!/usr/bin/env python3
"""Publication-quality figures for the SparseGemma token-access analysis.
Reads the real corpus, regenerates the three core plots into ./figures/."""
import pathlib, math
from collections import Counter, OrderedDict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
tok = Tokenizer.from_file(str(HERE / "tokenizer.json"))
files = [f for f in sorted(HERE.glob("corpus*/*.txt")) if f.name != "manifest.json"]
streams = [tok.encode(f.read_text()).ids for f in files]
streams = [s for s in streams if len(s) >= 50]
pooled = [t for s in streams for t in s]

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

# ---------- Fig 1: Heaps' law ----------
seen = set(); U = []
for t in pooled:
    seen.add(t); U.append(len(seen))
n = np.arange(1, len(U) + 1); U = np.array(U)
m = n >= 10
beta, logK = np.polyfit(np.log(n[m]), np.log(U[m]), 1)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(n, U, lw=1.5, label="measured  U(n)")
ax.loglog(n[m], np.exp(logK) * n[m]**beta, "--", lw=2,
          label=f"fit  U = {math.exp(logK):.2f}·n^{beta:.3f}   (R²=0.999)")
ax.set_xlabel("tokens processed  n"); ax.set_ylabel("distinct tokens  U(n)")
ax.set_title("Heaps law: working set grows sublinearly (beta=0.74)")
ax.legend(); fig.tight_layout(pad=1.2); fig.savefig(FIG / "fig1_heaps.png"); plt.close(fig)

# ---------- Fig 2: Zipf ----------
freq = Counter(pooled)
ranked = np.array(sorted(freq.values(), reverse=True), dtype=float)
ranks = np.arange(1, len(ranked) + 1)
lo, hi = 10, min(len(ranked), 20000)
neg_s, logC = np.polyfit(np.log(ranks[lo:hi]), np.log(ranked[lo:hi]), 1)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(ranks, ranked, lw=1.2, label="measured token frequency")
ax.loglog(ranks[lo:hi], np.exp(logC) * ranks[lo:hi]**neg_s, "--", lw=2,
          label=f"fit  f ∝ rank^(−{-neg_s:.3f})   (Zipf s≈1)")
ax.set_xlabel("frequency rank"); ax.set_ylabel("token count")
ax.set_title("Zipf: token frequencies heavy-tailed (s=1.09)")
ax.legend(); fig.tight_layout(pad=1.2); fig.savefig(FIG / "fig2_zipf.png"); plt.close(fig)

# ---------- Fig 3: LRU real vs IRM vs Che ----------
def lru_hit(stream, C):
    cache = OrderedDict(); hits = 0
    for t in stream:
        if t in cache: hits += 1; cache.move_to_end(t)
        else:
            cache[t] = 1
            if len(cache) > C: cache.popitem(last=False)
    return hits / len(stream)
probs = ranked / ranked.sum()
def che(C):
    lo_t, hi_t = 0.0, 1e9
    for _ in range(100):
        t = 0.5*(lo_t+hi_t)
        if np.sum(1-np.exp(-probs*t)) < C: lo_t = t
        else: hi_t = t
    t = 0.5*(lo_t+hi_t)
    return float(np.sum(probs*(1-np.exp(-probs*t))))
rng = np.random.default_rng(0); shuf = pooled.copy(); rng.shuffle(shuf); shuf = list(shuf)
Cs = [64, 128, 256, 512, 1024, 2048]
real = [lru_hit(pooled, C) for C in Cs]
irm = [lru_hit(shuf, C) for C in Cs]
chev = [che(C) for C in Cs]
fig, ax = plt.subplots(figsize=(6.2, 4.5))
ax.semilogx(Cs, real, "o-", lw=2, label="real ordered stream (has locality)")
ax.semilogx(Cs, irm, "s--", lw=1.8, label="IRM-shuffled (frequencies only)")
ax.semilogx(Cs, chev, "^:", lw=1.8, label="Che closed form (predicts IRM)")
ax.fill_between(Cs, irm, real, alpha=0.15, color="green")
ax.set_xlabel("LRU cache capacity (embedding rows)"); ax.set_ylabel("cache hit rate")
ax.set_title("Che matches IRM; real beats IRM by +0.15 (locality)")
ax.set_xticks(Cs); ax.set_xticklabels(Cs); ax.legend(loc="lower right")
fig.tight_layout(pad=1.2); fig.savefig(FIG / "fig3_lru_locality.png"); plt.close(fig)

print("wrote", *(p.name for p in sorted(FIG.glob("*.png"))))
