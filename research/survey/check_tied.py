import onnx
# Critical check: is the OUTPUT projection (logits) a SEPARATE lm_head tensor, or does it reuse the
# embed_tokens table (tied weights)? If tied, the output side needs the full table for logits over vocab.
for name, path in [("Qwen2.5-0.5B","Qwen2.5-0.5B.onnx"),
                   ("SmolLM2-360M","SmolLM2-360M.onnx"),
                   ("Gemma-3-270M","Gemma-3-270M.onnx")]:
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    emb_names = {i.name for i in g.initializer if len(i.dims)==2 and i.dims[0]>=30000}
    print("\n=== %s ===" % name)
    print("  vocab-sized tensors:", emb_names)
    consumers = {}
    for n in g.node:
        for inp in n.input:
            if inp in emb_names:
                consumers.setdefault(inp, []).append(n.op_type)
    for t, ops in consumers.items():
        gather = [o for o in ops if 'Gather' in o]
        matmul = [o for o in ops if o in ('MatMul','Gemm','MatMulNBits')]
        print("  tensor '%s' consumed by: %s" % (t, ops))
        if gather and matmul:
            print("    >>> TIED: BOTH input lookup %s AND output projection %s" % (gather, matmul))
        elif gather:
            print("    >>> input-lookup only (output head is SEPARATE)")
        elif matmul:
            print("    >>> output-projection only")
    heads = [i.name for i in g.initializer if 'head' in i.name.lower()]
    print("  head-named tensors:", heads[:4] if heads else 'none')
