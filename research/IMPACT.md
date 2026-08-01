# Impact & economics of token-sparse embedding loading

Honest, per-unit arithmetic. We do **not** fabricate absolute adoption numbers — figures are stated as
rates ("per million fresh sessions") so anyone can multiply by a defensible session count. Assumptions
are labeled and conservative-to-midrange.

## Per-session bytes avoided (input-side embedding, fresh/uncached session)

| Model | Embedding avoided | Full download → after |
|---|---|---|
| Qwen2.5-0.5B | **271 MB** | 483 → ~212 MB (−56%) |
| SmolLM2-360M | 94 MB | 273 → ~179 MB (−34%) |
| Gemma-3-270M | 84 MB | 273 → ~189 MB (−31%) |

(“after” = model minus embedding table; the actual per-session embedding *fetched* is <1 MB by Heaps'
law, so the avoided amount is essentially the whole table on first use, and ~100% on every cached use.)

## Aggregate, per 1,000,000 fresh sessions

Assumptions: CDN egress **$0.06/GB** (mid-range; Cloudflare/Fastly/AWS span ~$0.02–0.09), data-transfer
energy **0.06 kWh/GB** (order-of-magnitude; literature ~0.03–0.1), grid **0.4 kg CO₂/kWh** (global avg).

| Model | Traffic saved | CDN cost saved | CO₂ avoided |
|---|---|---|---|
| Qwen2.5-0.5B | **265 TB** | ~$15,900 | ~6.4 t |
| SmolLM2-360M | 91 TB | ~$5,500 | ~2.2 t |
| Gemma-3-270M | 82 TB | ~$4,900 | ~2.0 t |

Browser-LLM apps already run at millions-of-sessions scale (HF hosts these models; Transformers.js is
widely embedded). At 10 M sessions the Qwen row alone is **~2.6 PB / ~$160k / ~64 t CO₂** avoided — from
one opt-in loader flag, with no quality change.

## Who benefits

- **Developers / app makers:** faster first-load, lower bounce, works on slow links and metered data —
  the difference between "usable on 4G" and "not."
- **Users in low-bandwidth regions:** the group most excluded by a 270 MB download is the one this helps
  most. This is the accessibility story, not a footnote.
- **Model hosts / CDNs (HuggingFace, etc.):** direct egress-cost and infrastructure reduction.
- **The planet:** transfer energy scales with bytes moved; moving 30–56% fewer bytes per session is a
  real, if modest-per-session, aggregate reduction.

## Business model — the honest read

This technique's *highest-value use is open-source*, not a product. Adoption (a merged Transformers.js /
onnxruntime-web feature) creates ecosystem-wide impact and durable reputational capital; gatekeeping it
would cap adoption and defeat the purpose. Real adjacent commercial surfaces, in honest priority order:

1. **Reputation → opportunity (the real asset).** Being the person who measured an ecosystem-wide
   inefficiency and shipped the fix, upstreamed into the standard tools, is career-defining leverage
   (hiring, advisory, founding credibility) worth far more than licensing the trick.
2. **On-device / edge-AI consulting & SDK.** Teams shipping private in-browser or on-device LLMs (health,
   finance, gov — exactly SparseGemma's demo domain) will pay for help getting model delivery small and
   fast; this technique plus the draft-oracle prefetch is a concrete, saleable capability.
3. **A "sparse model delivery" layer / CDN add-on.** A hosted service that serves any ONNX LLM with
   range-fetchable embeddings + edge caching + prefetch. Thin margin, real only at scale, and only after
   the open technique is proven and adopted.

Recommendation: **ship it open, get it upstreamed, let the reputation compound.** Monetize the *expertise*
(2), not the *trick* (which should be free and everywhere).
