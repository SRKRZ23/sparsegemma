#!/usr/bin/env python3
"""Generate a diverse corpus of realistic multi-turn conversations via the Gemma/Gemini API,
to empirically measure token-access statistics (Heaps' law, Zipf's law) for on-device LLM serving.

Diversity is deliberate: we want to show the sublinear-unique-token law is not an artifact of one
narrow domain. Output: one .txt per conversation in ./corpus/, plus a manifest.
"""
import os, json, time, urllib.request, urllib.error, pathlib

API_KEY = os.environ["GOOGLE_API_KEY"]
# Round-robin across models — each has its own per-minute free-tier request quota, so cycling
# multiplies effective throughput and spreads load. All produce representative conversational text.
MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemma-4-31b-it"]
def url_for(model):
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"

OUT = pathlib.Path(__file__).parent / "corpus"
OUT.mkdir(exist_ok=True)

# Deliberately diverse conversation seeds across domains — health, coding, cooking, travel, legal,
# science, casual chat, finance, education, creative — so the measured law generalizes.
SEEDS = [
    "Write a realistic 8-turn back-and-forth conversation between a worried patient and a health-navigation assistant about persistent headaches. Include the assistant's full detailed replies.",
    "Write a realistic 8-turn conversation between a beginner programmer and a coding assistant debugging a Python async bug. Full replies.",
    "Write a realistic multi-turn conversation about planning a two-week trip to Japan on a budget. Full detailed assistant replies.",
    "Write a realistic multi-turn conversation where someone asks a legal-info assistant about tenant rights during an eviction. Full replies (general info, not legal advice).",
    "Write a realistic multi-turn conversation explaining quantum entanglement to a curious high-schooler, with follow-up questions. Full replies.",
    "Write a realistic multi-turn cooking conversation troubleshooting why homemade bread isn't rising, with detailed replies.",
    "Write a realistic multi-turn conversation about personal budgeting and paying off credit card debt. Full replies.",
    "Write a realistic multi-turn conversation between a student and a tutor working through calculus derivatives step by step.",
    "Write a realistic casual multi-turn chat about recommending science fiction books and discussing themes.",
    "Write a realistic multi-turn conversation about training for a first marathon, covering schedule, nutrition, injury prevention.",
    "Write a realistic multi-turn conversation diagnosing why a houseplant's leaves are turning yellow, with detailed care advice.",
    "Write a realistic multi-turn conversation about starting a small vegetable garden in a cold climate.",
    "Write a realistic multi-turn conversation explaining how mortgages and interest rates work to a first-time homebuyer.",
    "Write a realistic multi-turn conversation about learning to play guitar as an adult beginner.",
    "Write a realistic multi-turn conversation troubleshooting slow home wifi with a tech-support assistant.",
    "Write a realistic multi-turn conversation about meal-prepping healthy lunches for a busy work week.",
    "Write a realistic multi-turn conversation explaining climate change feedback loops to a skeptical relative.",
    "Write a realistic multi-turn conversation about resolving a conflict with a difficult coworker.",
    "Write a realistic multi-turn conversation planning a child's birthday party on a tight budget.",
    "Write a realistic multi-turn conversation about understanding and improving one's credit score.",
]

_model_i = [0]
def call(prompt, retries=8):
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 4096},
    }).encode()
    for attempt in range(retries):
        model = MODELS[_model_i[0] % len(MODELS)]
        _model_i[0] += 1  # rotate model every attempt (spreads across quotas)
        try:
            req = urllib.request.Request(url_for(model), data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            return d["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # honor the server's suggested retry delay if present, else exponential
                wait = min(30, 4 * (attempt + 1))
                print(f"  {model} 429 rate-limited, wait {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"  {model} HTTP {e.code}, next model", flush=True)
                time.sleep(1)
        except Exception as e:
            print(f"  {model} err {e}, retry", flush=True)
            time.sleep(2)
    return None

manifest = []
# Generate multiple variations per seed for a larger, richer corpus.
VARIATIONS = 4
idx = 0
for seed in SEEDS:
    for v in range(VARIATIONS):
        idx += 1
        out_path = OUT / f"conv_{idx:04d}.txt"
        if out_path.exists():
            manifest.append(out_path.name)
            continue
        vary = f" (variation {v+1}: use a different specific scenario, names, and details than a typical example)"
        text = call(seed + vary)
        if text:
            out_path.write_text(text)
            manifest.append(out_path.name)
            print(f"[{idx}] {len(text)} chars -> {out_path.name}", flush=True)
        else:
            print(f"[{idx}] FAILED", flush=True)
        time.sleep(3.5)  # ~17 req/min, safely under the 20/min free-tier per-model limit

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nDone: {len(manifest)} conversations in {OUT}")
