import os, json, time, urllib.request, urllib.error, pathlib
KEY = os.environ["GOOGLE_API_KEY"]
OUT = pathlib.Path("corpus_chat"); OUT.mkdir(exist_ok=True)
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={KEY}"
SEEDS = [
 "Write a realistic 10-turn chat between a worried patient and a health assistant about chest tightness. Full replies.",
 "Write a realistic 10-turn chat debugging a Python KeyError with a coding assistant. Full replies.",
 "Write a realistic 10-turn chat planning a budget trip to Portugal. Full replies.",
 "Write a realistic 10-turn chat about fixing a sourdough starter that won't bubble. Full replies.",
 "Write a realistic 10-turn chat explaining compound interest to a teenager. Full replies.",
 "Write a realistic 10-turn chat troubleshooting a car that won't start on cold mornings. Full replies.",
 "Write a realistic 10-turn chat about beginning meditation for anxiety. Full replies.",
 "Write a realistic 10-turn chat choosing a first programming language for a career switch. Full replies.",
 "Write a realistic 10-turn chat about caring for a new puppy's first week. Full replies.",
 "Write a realistic 10-turn chat explaining how vaccines train the immune system. Full replies.",
 "Write a realistic 10-turn chat planning a vegetable garden for a small balcony. Full replies.",
 "Write a realistic 10-turn chat about negotiating a salary offer. Full replies.",
 "Write a realistic 10-turn chat troubleshooting why a laptop overheats. Full replies.",
 "Write a realistic 10-turn chat about training for a 10k run as a beginner. Full replies.",
 "Write a realistic 10-turn chat explaining mortgage basics to a first-time buyer. Full replies.",
]
for i, seed in enumerate(SEEDS, 1):
    out = OUT / f"chat_{i:03d}.txt"
    if out.exists() and out.stat().st_size > 500: continue
    body = json.dumps({"contents":[{"parts":[{"text":seed}]}],
        "generationConfig":{"temperature":1.0,"maxOutputTokens":4096}}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            text = d["candidates"][0]["content"]["parts"][0]["text"]
            out.write_text(text); print(f"[{i}] {len(text)} chars", flush=True)
            break
        except urllib.error.HTTPError as e:
            w = 20 if e.code==429 else 5
            print(f"[{i}] HTTP {e.code}, wait {w}s", flush=True); time.sleep(w)
        except Exception as e:
            print(f"[{i}] {e}", flush=True); time.sleep(5)
    time.sleep(12)  # very gentle
print("done:", len(list(OUT.glob('chat_*.txt'))))
