#!/usr/bin/env python3
"""Extend the dump with config scalars, embedding and head slices."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = '        _t.save(blob, "/workspace/logs/atom_params.pt")'
new = ('        blob["cfg"] = {\n'
       '            "eps": float(eps), "head_dim": int(D), "nH": int(nH), "nKV": int(nKV),\n'
       '            "n_experts": int(n_experts), "top_k": int(top_k),\n'
       '            "sliding_window": None if W is None else int(W),\n'
       '        }\n'
       '        blob["embed"] = model.model.embed_tokens.weight[:64].detach().cpu()\n'
       '        blob["final_norm"] = model.model.norm.weight.detach().cpu()\n'
       '        blob["lm_head"] = model.lm_head.weight[:64].detach().cpu()\n'
       '        _t.save(blob, "/workspace/logs/atom_params.pt")')
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("dump extended")
