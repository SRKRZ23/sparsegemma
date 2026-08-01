import onnx, os
for name, total in [("Qwen2.5-0.5B", 483003582), ("SmolLM2-360M", 272737275)]:
    m = onnx.load(f"{name}.onnx", load_external_data=False)
    # find biggest 2D vocab-sized initializer and its raw byte size (inline raw_data)
    best = None
    for init in m.graph.initializer:
        dims = list(init.dims)
        if len(dims) == 2 and dims[0] >= 30000:
            nbytes = len(init.raw_data)
            if best is None or nbytes > best[2]:
                best = (init.name, dims, nbytes, init.data_type)
    if best:
        nm, dims, nbytes, dt = best
        dtype = {1:"float32",10:"float16",2:"uint8",3:"int8",21:"uint4",22:"int4"}.get(dt, f"type{dt}")
        # also: is the LM head tied? check for a MatMul using this same tensor
        print(f"{name}: embed tensor '{nm}' dims={dims} dtype={dtype}")
        print(f"   raw bytes={nbytes/1e6:.1f} MB  =>  {100*nbytes/total:.1f}% of the {total/1e6:.0f} MB single-file model")
