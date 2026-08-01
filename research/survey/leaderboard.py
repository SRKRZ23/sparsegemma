#!/usr/bin/env python3
"""Scalable survey: for many popular browser-deployable ONNX LLMs, compute what fraction of the model
download is the token-embedding table — WITHOUT downloading the multi-GB weights.

Method: fetch config.json (tiny) for vocab_size / hidden_size / tie_word_embeddings, and HEAD the ONNX
file(s) for total bytes. Embedding bytes = vocab * hidden * bytes_per_element for the given dtype.
Validated against the 3 models measured exactly (Qwen 56.4%, SmolLM2 34.6%, Gemma-3-270M 30.8%)."""
import json, urllib.request, urllib.error

# (display name, repo, onnx subpath, dtype-bytes-per-element for the embedding in this quant variant)
# q4f16 exports typically keep embeddings fp16 (2 B) unless the op is GatherBlockQuantized (~0.6 B eff).
MODELS = [
    ("Qwen2.5-0.5B-Instruct", "onnx-community/Qwen2.5-0.5B-Instruct", "onnx/model_q4f16.onnx", 2.0),
    ("Qwen2.5-1.5B-Instruct", "onnx-community/Qwen2.5-1.5B-Instruct", "onnx/model_q4f16.onnx", 2.0),
    ("SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "onnx/model_q4f16.onnx", 2.0),
    ("SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "onnx/model_q4f16.onnx", 2.0),
    ("Llama-3.2-1B-Instruct", "onnx-community/Llama-3.2-1B-Instruct", "onnx/model_q4f16.onnx", 2.0),
    ("Phi-3.5-mini-instruct", "onnx-community/Phi-3.5-mini-instruct-onnx-web", "onnx/model_q4f16.onnx", 2.0),
    ("gemma-3-270m-it", "onnx-community/gemma-3-270m-it-ONNX", "onnx/model_q4f16.onnx", 0.56),
    ("TinyLlama-1.1B-Chat", "onnx-community/TinyLlama-1.1B-Chat-v1.0", "onnx/model_q4f16.onnx", 2.0),
    ("Qwen2.5-Coder-0.5B", "onnx-community/Qwen2.5-Coder-0.5B-Instruct", "onnx/model_q4f16.onnx", 2.0),
    ("Llama-3.2-3B-Instruct", "onnx-community/Llama-3.2-3B-Instruct", "onnx/model_q4f16.onnx", 2.0),
    ("Qwen3-0.6B", "onnx-community/Qwen3-0.6B-ONNX", "onnx/model_q4f16.onnx", 2.0),
    ("Qwen3-1.7B", "onnx-community/Qwen3-1.7B-ONNX", "onnx/model_q4f16.onnx", 2.0),
]

def get_json(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sparsegemma-survey/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception:
        return None

def head_size(url):
    for u in (url,):
        try:
            req = urllib.request.Request(u, method="HEAD", headers={"User-Agent": "sparsegemma-survey/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                cl = int(r.headers.get("content-length", 0))
                if cl > 0:
                    return cl
        except urllib.error.HTTPError as e:
            # HF often 302s HEAD; follow via GET Range 0-0 to read Content-Range total
            pass
        # fallback: Range 0-0 to get total from Content-Range
        try:
            req = urllib.request.Request(u, headers={"Range": "bytes=0-0", "User-Agent": "sparsegemma-survey/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                cr = r.headers.get("content-range", "")
                if "/" in cr:
                    return int(cr.split("/")[-1])
        except Exception:
            return 0
    return 0

rows = []
for name, repo, path, bpe in MODELS:
    cfg = get_json(f"https://huggingface.co/{repo}/resolve/main/config.json")
    if not cfg:
        print(f"{name:26s}  config not found ({repo})"); continue
    vocab = cfg.get("vocab_size")
    hidden = cfg.get("hidden_size")
    tied = cfg.get("tie_word_embeddings")
    if not vocab or not hidden:
        print(f"{name:26s}  missing vocab/hidden in config"); continue
    onnx_url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    data_url = onnx_url + "_data"
    graph = head_size(onnx_url)
    data = head_size(data_url)
    total = graph + data if data else graph
    emb_bytes = vocab * hidden * bpe
    pct = 100 * emb_bytes / total if total else float("nan")
    rows.append((name, vocab, hidden, tied, emb_bytes, total, pct))
    print(f"{name:26s} vocab={vocab:>7} hidden={hidden:>5} tied={str(tied):>5} "
          f"emb={emb_bytes/1e6:6.0f}MB total={total/1e6:6.0f}MB  => {pct:5.1f}% of download")

rows.sort(key=lambda r: -r[6])
with open("leaderboard.json", "w") as f:
    json.dump([{"model": r[0], "vocab": r[1], "hidden": r[2], "tied": r[3],
                "emb_mb": round(r[4]/1e6, 1), "total_mb": round(r[5]/1e6, 1),
                "emb_pct": round(r[6], 1)} for r in rows], f, indent=2)
print(f"\nwrote leaderboard.json ({len(rows)} models)")
