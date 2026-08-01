import onnx, urllib.request, sys

MODELS = {
  "Qwen2.5-0.5B":  "onnx-community/Qwen2.5-0.5B-Instruct/resolve/main/onnx/model_q4f16.onnx",
  "Gemma-3-270M":  "onnx-community/gemma-3-270m-it-ONNX/resolve/main/onnx/model_q4f16.onnx",
  "SmolLM2-360M":  "HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/onnx/model_q4f16.onnx",
}

def data_url_size(model_path):
    # the .onnx_data blob sits next to the .onnx graph
    data_url = "https://huggingface.co/" + model_path + "_data"
    try:
        req = urllib.request.Request(data_url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("content-length", 0)), data_url
    except Exception:
        return None, data_url

for name, path in MODELS.items():
    print(f"\n=== {name} ===")
    url = "https://huggingface.co/" + path
    try:
        urllib.request.urlretrieve(url, f"{name}.onnx")
        m = onnx.load(f"{name}.onnx", load_external_data=False)
    except Exception as e:
        print("  load err:", e); continue
    total_data, data_url = data_url_size(path)
    # find embedding tensor: largest 2-D initializer whose first dim is the vocab (large), OR the
    # quantized weight feeding a Gather/GatherBlockQuantized producing inputs_embeds
    emb = None
    for init in m.graph.initializer:
        dims = list(init.dims)
        if len(dims) == 2 and dims[0] >= 30000:  # vocab-sized first dim
            ext = {e.key: e.value for e in init.external_data}
            length = int(ext.get("length", 0))
            if emb is None or length > emb[3]:
                emb = (init.name, dims, ext.get("offset"), length)
    # also detect the op that does the embedding lookup
    ops = set(n.op_type for n in m.graph.node)
    gather_ops = [o for o in ops if "Gather" in o]
    if emb:
        name_e, dims, off, length = emb
        pct = 100*length/total_data if total_data else float('nan')
        print(f"  embedding tensor: {name_e}")
        print(f"  dims={dims}  external_data length={length/1e6:.1f} MB  offset={off}")
        print(f"  total .onnx_data={total_data/1e6:.1f} MB" if total_data else "  total unknown")
        print(f"  >>> embedding = {pct:.1f}% of the model download")
        print(f"  lookup op(s): {gather_ops}  (row-major external-data => Range-fetchable: "
              f"{'YES' if off is not None else 'no'})")
    else:
        print(f"  no vocab-sized 2-D initializer found; ops={sorted(ops)[:12]}")
