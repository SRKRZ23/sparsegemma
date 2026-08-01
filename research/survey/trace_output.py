import onnx
# Trace whether embed_tokens.weight feeds the final logits via Transpose->MatMul (tied output head).
for name, path in [("Qwen2.5-0.5B","Qwen2.5-0.5B.onnx"), ("SmolLM2-360M","SmolLM2-360M.onnx")]:
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    emb = [i.name for i in g.initializer if len(i.dims)==2 and i.dims[0]>=30000][0]
    # follow the dataflow from emb through Transpose to a MatMul that reaches a graph output
    outputs = {o.name for o in g.output}
    # build producer map
    node_by_out = {}
    for n in g.node:
        for o in n.output:
            node_by_out[o] = n
    # find Transpose consuming emb
    trans = [n for n in g.node if n.op_type=="Transpose" and emb in n.input]
    print("\n=== %s ===" % name)
    print("  emb tensor:", emb)
    for tn in trans:
        to = tn.output[0]
        # find MatMul consuming the transpose output
        mm = [n for n in g.node if n.op_type in ("MatMul","Gemm") and to in n.input]
        for mnode in mm:
            mo = mnode.output[0]
            reaches = mo in outputs
            print("  emb -> Transpose -> MatMul '%s' -> output '%s'  (graph logits output: %s)"
                  % (mnode.name, mo, reaches or ('logits' in mo.lower())))
            print("  >>> TIED OUTPUT HEAD: the full embedding table is ALSO the output projection.")
