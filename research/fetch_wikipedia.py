#!/usr/bin/env python3
"""Fetch diverse real human text from Wikipedia (public API, no key, not rate-limited) as a corpus
for measuring token-access statistics. Real human prose is the gold standard for demonstrating the
universal linguistic laws (Heaps, Zipf); we deliberately span many domains so the measured exponents
are not an artifact of one topic. Each article -> one .txt in ./corpus_wiki/.
"""
import json, time, urllib.request, urllib.parse, pathlib

OUT = pathlib.Path(__file__).parent / "corpus_wiki"
OUT.mkdir(exist_ok=True)

# Deliberately diverse: science, history, medicine, tech, arts, geography, sports, food, law, biography.
TITLES = [
    "Immune system", "Photosynthesis", "Black hole", "French Revolution", "Great Barrier Reef",
    "Machine learning", "Antibiotic", "Roman Empire", "Climate change", "Jazz",
    "Quantum mechanics", "Volcano", "Democracy", "Coffee", "Human brain",
    "Renewable energy", "Ancient Egypt", "Vaccine", "Basketball", "Mount Everest",
    "Artificial intelligence", "Dinosaur", "Ocean current", "Symphony", "Blood",
    "Electric car", "Silk Road", "Diabetes", "Chess", "Amazon rainforest",
    "Cryptography", "Antarctica", "Vitamin", "Guitar", "Nervous system",
    "Solar System", "Industrial Revolution", "Coral", "Marathon", "Tea",
    "Neural network", "Earthquake", "Constitution", "Bread", "Heart",
    "Space exploration", "Byzantine Empire", "Influenza", "Football", "Pacific Ocean",
]

def fetch(title):
    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "titles": title, "redirects": "1",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "sparsegemma-research/1.0 (research measurement)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    pages = d["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")

manifest = []
for i, title in enumerate(TITLES, 1):
    out = OUT / f"wiki_{i:03d}.txt"
    if out.exists() and out.stat().st_size > 500:
        manifest.append(out.name); continue
    try:
        text = fetch(title)
        if text and len(text) > 500:
            out.write_text(text)
            manifest.append(out.name)
            print(f"[{i}] {title}: {len(text)} chars", flush=True)
        else:
            print(f"[{i}] {title}: too short/empty", flush=True)
    except Exception as e:
        print(f"[{i}] {title}: ERR {e}", flush=True)
    time.sleep(0.3)

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nDone: {len(manifest)} articles in {OUT}")
