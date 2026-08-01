# PocketTriage — private, on-device health navigation (Gemma 4)

Built for **Build with Gemma NYC: On-Device AI for Healthcare** (Kaggle/AI Tinkerers hackathon, 2026-08-01),
Track 3 — *On-Device Private Health*.

**⚠️ This project is not affiliated with or eligible for the hackathon's prizes/judging** — the event is
restricted to in-person NYC builders (confirmed by the host in the event's own Q&A discussion). It was
built anyway as a public speed-build exercise, deliberately following the hackathon's brief and rubric for
practice/portfolio value, not as a prize submission.

## What it is
A single-page web app that runs **Gemma 4 E2B** (`onnx-community/gemma-4-E2B-it-ONNX`, q4f16 quantized)
**entirely inside the browser** via [Transformers.js](https://huggingface.co/docs/transformers.js) and
WebGPU — no backend, no API calls after the one-time model download. Nothing a user types is ever
transmitted anywhere.

**Scope, per the hackathon's own rules**: decision-support and education only — explains general urgency
language (the kind hospital triage systems like ESI reason about) and what to ask a real clinician. It
does **not** diagnose or treat, and the UI carries an explicit disclaimer to that effect. No real patient
data is used or requested anywhere — general/synthetic descriptions only.

## Why Gemma 4 specifically
The E2B ONNX build is small enough to run client-side at usable speed via WebGPU, and Transformers.js
ships a documented `Gemma4ForConditionalGeneration` class for it — this is genuinely new (the model and
this browser packaging both landed within the last few weeks as of this build), which is also why this
repo is honest about not having been able to fully verify long-running interactive sessions end-to-end in
every browser.

## Architecture
```
index.html + app.js (vanilla JS, ES modules, no build step)
        │
        ▼
Transformers.js (CDN import) ── AutoProcessor + Gemma4ForConditionalGeneration.from_pretrained(
        │                          model_id, { dtype: "q4f16", device: "webgpu" })
        ▼
WebGPU (browser's own GPU access) ── all inference happens here, on the user's device
        │
        ▼
Streamed response in the chat UI (TextStreamer, token-by-token)
```

## Honest status
- Code is real, deployed, and passes a Node.js syntax check.
- **Live WebGPU inference was NOT verified in the build environment** (no GPU/WebGPU-capable browser
  available there) — this needs to be confirmed in a real WebGPU browser (recent Chrome/Edge) before
  relying on it. Model class (`Gemma4ForConditionalGeneration`) and the chat-template call pattern follow
  the model card's own documented usage, but Gemma 4's Transformers.js support is only days old at time of
  writing, so some API drift is possible.
- No image/audio modality wired up (the underlying model supports it) — text chat only, kept in scope for
  a same-day build.

## Run locally
```bash
cd public && python3 -m http.server 8834
# open http://localhost:8834 in a WebGPU-capable browser (Chrome/Edge, recent version)
```

## Live demo
https://pockettriage-gemma4.netlify.app (WebGPU browser required)
