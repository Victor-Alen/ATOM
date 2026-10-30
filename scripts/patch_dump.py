#!/usr/bin/env python3
"""Dump ATOM's layer parameters once, for an offline bitwise comparison with HF."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
if "ATOM_REFERENCE_DUMP" not in s:
    anchor = "    for li, layer in enumerate(model.model.layers):"
    dump = '''    if os.environ.get("ATOM_REFERENCE_DUMP") == "1" and "dumped" not in _ROT:
        import torch as _t

        blob = {}
        for li in (0, 2):
            lay = model.model.layers[li]
            a = lay.self_attn
            q_w, k_w, v_w = torch.split(a.qkv_proj.weight, [nH * D, nKV * D, nKV * D], dim=0)
            blob[f"L{li}.q"] = q_w.detach().cpu()
            blob[f"L{li}.k"] = k_w.detach().cpu()
            blob[f"L{li}.v"] = v_w.detach().cpu()
            blob[f"L{li}.o"] = a.o_proj.weight.detach().cpu()
            blob[f"L{li}.q_norm"] = a.q_norm.weight.detach().cpu()
            blob[f"L{li}.k_norm"] = a.k_norm.weight.detach().cpu()
            blob[f"L{li}.in_ln"] = lay.input_layernorm.weight.detach().cpu()
            blob[f"L{li}.post_ln"] = lay.post_attention_layernorm.weight.detach().cpu()
            blob[f"L{li}.gate"] = lay.mlp.gate.weight.detach().cpu()
            blob[f"L{li}.w13"] = _unshuffle(lay.mlp.experts.w13_weight).detach().cpu()
            blob[f"L{li}.w2"] = _unshuffle(lay.mlp.experts.w2_weight).detach().cpu()
        _t.save(blob, "/workspace/logs/atom_params.pt")
        _ROT["dumped"] = True
        print("[ref] dumped ATOM params -> /workspace/logs/atom_params.pt", flush=True)

'''
    assert anchor in s
    s = s.replace(anchor, dump + anchor, 1)
    p.write_text(s)
    ast.parse(s)
    print("dump hook added")
else:
    print("already present")
