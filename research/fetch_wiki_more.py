import json, time, urllib.request, urllib.parse, pathlib
OUT = pathlib.Path("corpus_wiki"); OUT.mkdir(exist_ok=True)
TITLES = ["Vitamin","Guitar","Nervous system","Solar System","Industrial Revolution","Coral",
 "Marathon","Tea","Neural network","Earthquake","Constitution","Bread","Heart","Space exploration",
 "Byzantine Empire","Influenza","Football","Pacific Ocean","Genetics","Electricity","Rome","Piano",
 "Kidney","Wind power","Maya civilization","Cancer","Tennis","Rice","Copyright","Sahara",
 "Blockchain","Glacier","Federalism","Cheese","Lung","Aviation","Mongol Empire","Malaria",
 "Cricket (sport)","Mediterranean Sea","Evolution","Magnetism","Ottoman Empire","Violin","Liver",
 "Nuclear power","Inca Empire","Tuberculosis","Rugby","Nile"]
existing = len(list(OUT.glob("wiki_*.txt")))
idx = existing
for title in TITLES:
    idx += 1
    out = OUT / f"wiki_{idx:03d}.txt"
    if out.exists() and out.stat().st_size > 500: continue
    for attempt in range(5):
        try:
            p = urllib.parse.urlencode({"action":"query","format":"json","prop":"extracts",
                "explaintext":"1","titles":title,"redirects":"1"})
            req = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{p}",
                headers={"User-Agent":"sparsegemma-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            page = next(iter(d["query"]["pages"].values()))
            text = page.get("extract","")
            if text and len(text) > 500:
                out.write_text(text); print(f"[{idx}] {title}: {len(text)}", flush=True)
            break
        except Exception as e:
            print(f"[{idx}] {title} attempt {attempt}: {e}", flush=True)
            time.sleep(5*(attempt+1))
    time.sleep(2.0)  # gentle
print("done, total files:", len(list(OUT.glob("wiki_*.txt"))))
