Title: PocketTriage — A Private, Fully On-Device Health-Navigation Companion
Subtitle: Running Gemma 4 E2B entirely in the browser via WebGPU — no server, no cloud, no data leaving the device.

## Note on eligibility
This event's host confirmed in the discussion thread that participation is limited to in-person NYC
builders. I'm not local, so this submission is offered purely as a public build exercise following the
brief and rubric — not as a prize entry.

## Motivation & Problem Statement
Health-navigation questions ("is this urgent?", "what should I ask my doctor?") are exactly the kind of
sensitive queries people are least comfortable typing into a cloud service. On-device inference removes
that tradeoff entirely: if the model never leaves the phone or laptop, there's no server log, no account,
no data-sharing agreement to trust — the privacy guarantee is architectural, not a policy promise.

## What it does
PocketTriage is a single-page web app. A user describes symptoms in plain language; Gemma 4 explains, in
plain terms, the general urgency category that description would typically fall under (the kind of
reasoning ED triage systems like ESI use), suggests 2-3 questions to bring to a real clinician, and always
closes with a reminder to seek real care and call emergency services for anything severe. It is explicitly
scoped as decision-support and education — not diagnosis — per this hackathon's own rules, and no real
patient data is used anywhere; only general/synthetic descriptions.

## Gemma 4 Integration
Model: `onnx-community/gemma-4-E2B-it-ONNX`, q4f16 quantization, loaded via Transformers.js's
`Gemma4ForConditionalGeneration.from_pretrained(model_id, { dtype: "q4f16", device: "webgpu" })`. Inference
runs on the user's own GPU through WebGPU — after the one-time model download, the app makes zero network
calls. Responses stream token-by-token via `TextStreamer` for a responsive chat feel.

## Architecture
```
index.html + app.js (vanilla JS, ES modules, zero build step)
   -> Transformers.js (CDN import): AutoProcessor + Gemma4ForConditionalGeneration
   -> WebGPU (the browser's own GPU access) — all inference happens here
   -> Streamed response in a minimal chat UI
```
No backend exists. The entire app is a static site (two files) deployable anywhere.

## Engineering challenges in this build window
- **Gemma 4's Transformers.js support is only days old.** The `Gemma4ForConditionalGeneration` class and
  its browser ONNX packaging landed very recently — meaning less community-tested surface area than an
  older model would have. I followed the model card's documented usage pattern exactly rather than
  guessing at API shape.
- **No GPU-equipped, WebGPU-capable browser was available in my own build/CI environment** to run a live
  end-to-end generation before writing this up. The code passes a Node.js syntax check and follows the
  documented API precisely, but I'm being explicit that full interactive-session verification still needs
  to happen in a real WebGPU browser (Chrome/Edge) — I'd rather disclose that honestly than claim a
  confirmed demo I couldn't actually run end-to-end myself in the time available.
- **Scope discipline under a hard time constraint.** The underlying E2B model supports image and audio
  input; I deliberately kept this build to text-only chat to ship a complete, honestly-scoped artifact
  rather than a half-working multimodal one.

## Privacy & Safety
No server exists to log anything. No account or login. No real patient data is requested or should be
entered — the UI banner says so explicitly. The system prompt hard-scopes the model to education/
navigation language and mandates a "seek real care" reminder on every response, in line with the
hackathon's decision-support-only, synthetic-data-only rule.

## Links
- Code: https://github.com/SRKRZ23/pockettriage-gemma4
- Live demo (requires a WebGPU-capable browser, e.g. recent Chrome/Edge): https://pockettriage-gemma4.netlify.app
