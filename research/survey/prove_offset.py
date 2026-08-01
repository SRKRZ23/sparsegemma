import onnx, mmap
# Prove the inline embedding IS range-fetchable: find the byte offset of its raw_data within the .onnx file.
name, total = "Qwen2.5-0.5B", 483003582
m = onnx.load(f"{name}.onnx", load_external_data=False)
emb = max((i for i in m.graph.initializer if len(i.dims)==2 and i.dims[0]>=30000),
          key=lambda i: len(i.raw_data))
raw = emb.raw_data
print(f"embedding '{emb.name}' dims={list(emb.dims)} raw={len(raw)/1e6:.1f}MB")
# locate a unique 64-byte signature from the MIDDLE of the tensor in the file
sig = raw[len(raw)//2 : len(raw)//2 + 64]
with open(f"{name}.onnx","rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = mm.find(sig)
    # the tensor start = pos - (len(raw)//2)
    tensor_start = pos - (len(raw)//2)
    # verify: the bytes at tensor_start for len(raw) equal raw
    mm.seek(tensor_start)
    check = mm[tensor_start:tensor_start+len(raw)]
    ok = (check == raw)
    print(f"signature found at file offset {pos}")
    print(f"=> embedding raw_data occupies bytes [{tensor_start}, {tensor_start+len(raw)}) of the .onnx file")
    print(f"=> VERIFIED contiguous & matches: {ok}")
    print(f"=> a browser can HTTP-Range-fetch just row r via bytes="
          f"{tensor_start}+r*{emb.dims[1]*2} .. +{emb.dims[1]*2-1}  (fp16, {emb.dims[1]} dims/row)")
    # per-row bytes for one token:
    row_bytes = emb.dims[1]*2
    print(f"=> ONE token embedding = {row_bytes} bytes; 400 unique tokens = {400*row_bytes/1024:.0f} KB "
          f"vs {len(raw)/1e6:.0f} MB full table ({100*400*row_bytes/len(raw):.3f}%)")
