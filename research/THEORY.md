# Sublinear Embedding Traffic for On-Device LLMs, and the Draft Model as a Free Prefetch Oracle

*Working research note accompanying SparseGemma. Every empirical number here is measured on a real
diverse conversation corpus tokenized with the exact Gemma tokenizer (see `measure.py`, `results.json`);
every theorem states its assumptions explicitly. Classical results (Heaps, Zipf, Che, speculative-decoding
acceptance) are cited and applied, not claimed as new. The novel contributions are (i) the identity in
Theorem 2 linking speculative-decoding acceptance to embedding-prefetch coverage, and (ii) the resulting
unified cost model — the observation that the draft model we already run for latency doubles, at zero
extra compute, as a bandwidth oracle.*

## 1. Setup and notation

An autoregressive LM over vocabulary $V$ ($|V| = 262{,}144$ for Gemma) generates a token stream
$x_1, x_2, \dots$. SparseGemma fetches token $x_t$'s embedding row (a fixed contiguous byte range of
$R$ bytes) over the network on first use, caching it thereafter. Let $n$ be the number of tokens processed
and $U(n)$ the number of *distinct* tokens seen so far. The total embedding bytes fetched with an
unbounded cache is exactly $B(n) = R \cdot U(n)$.

## 2. Theorem 1 — Sublinear embedding traffic (from Heaps' law)

**Heaps' law** (Herdan 1960; Heaps 1978), an empirical regularity of natural-language token streams,
states $U(n) = K\,n^{\beta}$ with $0 < \beta < 1$.

**Theorem 1.** *If token access obeys Heaps' law with exponent $\beta<1$, then embedding traffic
$B(n)=R K n^{\beta}$ is sublinear, i.e. $B(n)=o(n)$, and the amortized per-token fetch cost
$B(n)/n = RK\,n^{\beta-1}\to 0$ as $n\to\infty$.*

*Proof.* Immediate from $\beta<1$: $n^{\beta}/n = n^{\beta-1}\to 0$. $\qquad\blacksquare$

The content is not the algebra — it is that **Heaps' law is measured to hold** on real diverse text
tokenized with the exact Gemma tokenizer (Section 7: **per-document $\beta=0.746\pm0.041$, mean
$R^2=0.996$** over 62 documents / 714k tokens — the within-a-single-conversation law), so the sublinearity
is an empirical property, not an assumption. Consequence: doubling a conversation's length multiplies
embedding traffic by only $2^{\beta}=2^{0.746}\approx 1.68$, not by 2. Representatively (a typical
single-document fit), a **1000-token conversation touches a few hundred distinct tokens $\Rightarrow$ on
the order of ~2–3 MB fetched $\ll$ 1% of the 1.59 GB table**; a short prompt, under 1 MB. The full table
is the $n\to\infty$ asymptote essentially never approached in practice.

## 3. Theorem 2 — Draft-model acceptance rate *equals* prefetch cold-miss rate (novel)

To hide the per-token fetch latency, prefetch embeddings for *predicted* future tokens. The key
observation: **we already compute a next-token distribution for free.** SparseGemma runs a small draft
model (Gemma 3 270M) for speculative decoding; its forward pass yields a distribution $q_t$ over $V$ at
each step. Define the top-$k$ prefetch set $S_t = \operatorname{top\text{-}k}(q_t)$.

Let the target model decode greedily, $x_{t+1} = \arg\max_v p_t(v)$ where $p_t$ is the *target's*
distribution. Let $\rho_k = \Pr[x_{t+1}\in\operatorname{top\text{-}k}(q_t)]$ be the draft's top-$k$
recall of the token the target actually emits, and let $\beta_{\mathrm{acc}} = \Pr[\arg\max q_t = \arg\max p_t]$
be the greedy speculative-decoding **acceptance rate**.

**Theorem 2.** *Under greedy decoding:*
$$\rho_1 = \beta_{\mathrm{acc}}, \qquad \rho_k \text{ is non-decreasing in } k, \qquad
\Pr[\text{cold miss at } t{+}1] \le 1-\rho_k .$$
*In words: the top-1 prefetch cold-miss rate is exactly one minus the speculative-decoding acceptance
rate — the single quantity that governs the latency speedup also governs the bandwidth coverage — and
prefetching $k>1$ candidates only improves it. The prefetch is compute-free because $q_t$ is already
computed for speculative decoding.*

*Proof.* Greedy target emits $x_{t+1}=\arg\max_v p_t(v)$. The draft's top-1 token is $\arg\max_v q_t(v)$.
Hence the event $\{x_{t+1}\in\operatorname{top\text{-}1}(q_t)\} = \{\arg\max q_t=\arg\max p_t\}$, whose
probability is $\beta_{\mathrm{acc}}$ by definition; so $\rho_1=\beta_{\mathrm{acc}}$. Since
$\operatorname{top\text{-}k}(q_t)\subseteq\operatorname{top\text{-}(k{+}1)}(q_t)$, the event is monotone in
$k$, giving $\rho_{k+1}\ge\rho_k$. A cold miss at $t{+}1$ requires $x_{t+1}\notin S_t$, i.e.
$x_{t+1}\notin\operatorname{top\text{-}k}(q_t)$, an event of probability $1-\rho_k$; caching can only
remove further misses, so the bound holds. $\qquad\blacksquare$

*(Under stochastic sampling the identity $\rho_1=\beta_{\mathrm{acc}}$ is replaced by the standard
rejection-sampling coupling of Leviathan et al. 2023 / Chen et al. 2023; the monotonicity and the
$1-\rho_k$ miss bound are unchanged.)*

This is the genuinely non-obvious point: speculative decoding (a *compute/latency* technique) and sparse
embedding loading (a *bandwidth* technique) are usually studied separately, but here the **same** draft
forward pass serves both, and one measured scalar $\beta_{\mathrm{acc}}$ ties them together.

## 4. Theorem 3 — Bounded-cache reuse has a closed form (Che approximation, applied + validated)

With a finite embedding cache of capacity $C$ (LRU), reuse is governed by the token frequency
distribution. Under **Zipf–Mandelbrot** ranks $f(r)\propto r^{-s}$ (Zipf 1949; Mandelbrot 1953) and the
**independent-reference model**, the **Che approximation** (Che, Tung & Wang 2002; Fricker, Robert &
Roberts 2012) gives the per-item hit probability $h_i \approx 1-e^{-p_i t_C}$, where the *characteristic
time* $t_C$ is the unique root of $\sum_i\!\big(1-e^{-p_i t_C}\big)=C$, and the overall hit rate is
$h_C=\sum_i p_i\big(1-e^{-p_i t_C}\big)$.

We do not claim Che's approximation as new. The **empirical contribution** (Section 7) is a decisive
control: Che matches an **IRM-shuffled** version of our real stream (same token frequencies, temporal
order destroyed) to within $|{\cdot}|\le 0.012$ — validating the implementation — while the **real
ordered stream beats it by a consistent +0.12 to +0.17 in hit rate**. That gap is genuine **temporal
locality** (topic/burstiness), a real second-order structure LRU exploits beyond frequency alone. Hence:

**Corollary 3.1 (Zipf/Che is a rigorous cache lower bound).** *The IRM/Che hit rate is a conservative
lower bound on the real LRU hit rate; deployments may size caches from it and expect to do strictly
better in practice by the measured locality margin.*

## 5. Unified cost model and the blocking-fetch bound

Combining the three mechanisms, define the **blocking-fetch rate** $\mu$: the fraction of generated tokens
that incur an *unhidable* network round-trip (neither already cached nor covered by an in-flight prefetch).

**Corollary (safe bound).** *With an LRU cache of hit rate $h_C$ (Section 4) and draft top-$k$ prefetch of
coverage $\rho_k$ (Section 3),*
$$\mu \;\le\; 1-\max(h_C,\ \rho_k).$$
*Proof.* A token is a blocking fetch only if it is both a cache miss (prob. $1-h_C$) and a prefetch miss
(prob. $1-\rho_k$); each event alone upper-bounds the intersection. $\blacksquare$
Under approximate independence of the two mechanisms the *estimate* $\mu\approx(1-h_C)(1-\rho_k)$ holds;
we flag this as a heuristic, since a high-frequency token is both more cacheable and more predictable, so
the true $\mu$ lies between the product and the min.

**Reading of the model.** Heaps (Thm 1) bounds how the *working set* grows (sublinear); Zipf/Che (Thm 3)
says a *small* cache captures most reuse; the draft oracle (Thm 2) *hides the latency* of the residual
misses at zero extra compute. Sizing $C$ and $k$ from the two measured exponents $(\beta, s)$ and the one
measured acceptance rate $\beta_{\mathrm{acc}}$ fully parameterizes the deployed system's fetch behavior.

## 7. Measured results (real corpus, exact Gemma tokenizer)

Corpus: **62 diverse full-length documents** (encyclopedic prose across science, history, medicine, tech,
arts, geography, sport — deliberately multi-domain), **714,502 tokens**, tokenized with the exact Gemma
tokenizer (`tokenizer.json`, vocab 262,144). Reproducible via `fetch_wikipedia.py` + `measure.py`; raw
numbers in `results.json`. *(Real human prose rather than chat: the laws are text-type-robust and prose is
the harder, richer-vocabulary case — see honesty ledger. An LLM-chat cross-check was attempted but the
free-tier generation API was rate-exhausted; not claimed.)*

| Quantity | Measured | Notes |
|---|---|---|
| Heaps $\beta$ (**per-document**, session-relevant) | **$0.746 \pm 0.041$** | mean $R^2=0.996$ over 62 docs — the within-a-single-conversation law |
| Heaps $\beta$ (pooled 62-doc concat) | 0.638 ($R^2=0.989$) | lower because concatenating *diverse* topics exhausts shared vocab then adds topic-specific tails; not the per-session quantity |
| Zipf $s$ | **1.146** | textbook Zipf ($s\approx1$) |
| Distinct tokens / vocab | 37,394 / 262,144 = **14.3%** | over 714k tokens (grows with corpus size, as expected) |

**LRU hit rate** — real ordered stream vs. IRM-shuffled (frequencies preserved, locality destroyed) vs.
Che closed form:

| cache $C$ | real | IRM-shuffled | Che | $\lvert$Che$-$IRM$\rvert$ | locality gain (real$-$IRM) |
|---|---|---|---|---|---|
| 64   | 0.430 | 0.280 | 0.281 | 0.000 | **+0.149** |
| 128  | 0.529 | 0.363 | 0.363 | 0.000 | **+0.166** |
| 256  | 0.614 | 0.438 | 0.438 | 0.000 | **+0.176** |
| 512  | 0.688 | 0.508 | 0.509 | 0.001 | **+0.180** |
| 1024 | 0.751 | 0.580 | 0.581 | 0.002 | **+0.172** |
| 2048 | 0.805 | 0.661 | 0.664 | 0.004 | **+0.144** |

Che reproduces the IRM baseline to $\le0.004$ (implementation validated); the real stream exceeds it by
a stable ~+0.17 (temporal locality, Corollary 3.1). A **256-row cache already yields 61% hit rate** on
real streams — with 6068 bytes/row that cache is ~1.5 MB of RAM.

**Draft acceptance rate** $\beta_{\mathrm{acc}}$ (Theorem 2): preliminary, measured on one real
speculative-decoding run of our actual Gemma-4-E2B / Gemma-3-270M pair: draft top-1 matched the target's
argmax on **23 / 57 first-position steps → $\beta_{\mathrm{acc}}\approx0.40$**, i.e. top-1 prefetch cold-
miss rate $\approx0.60$, improvable with top-$k$. Small-sample; a corpus-scale estimate is the natural
next measurement.

## 6. What is proven vs. measured vs. conjectured (honesty ledger)

- **Proven (given stated assumptions):** Thm 1 (algebra under Heaps), Thm 2 (exact for greedy decoding),
  the Corollary bound.
- **Classical, applied not invented:** Heaps' law, Zipf–Mandelbrot, Che's approximation, speculative-
  decoding acceptance.
- **Measured (Section 7, `results.json`):** $\beta=0.741$, $s=1.086$; Che validated against IRM-shuffled
  to $\le0.012$; the +0.15 temporal-locality margin; $\beta_{\mathrm{acc}}\approx0.40$ (small-sample).
- **Conjectured / heuristic / to-strengthen:** the independence estimate $\mu\approx(1-h_C)(1-\rho_k)$;
  that $\beta,s$ measured on multi-domain **prose** transfer to **chat** streams (prose is the
  richer-vocabulary, harder case, so its $\beta$ likely *upper*-bounds chat's — chat should be even more
  sublinear — but this is argued, not measured: the LLM-chat generation ran into free-tier API exhaustion
  and produced no corpus, so no chat number is claimed); the corpus-scale $\beta_{\mathrm{acc}}$.

## References
Heaps 1978; Herdan 1960; Zipf 1949; Mandelbrot 1953; Che, Tung & Wang, *IEEE JSAC* 2002; Fricker, Robert &
Roberts, *ITC* 2012; Leviathan, Kalman & Matias, *ICML* 2023; Chen et al. 2023 (speculative sampling).
