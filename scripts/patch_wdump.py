#!/usr/bin/env python3
"""Dump the first row of the split q/k/v weights so they can be compared with the
checkpoint (a shuffled/strided layout would break my splitting)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        q_w, k_w, v_w = torch.split(attn.qkv_proj.weight, [nH * D, nKV * D, nKV * D], dim=0)"""
new = """        q_w, k_w, v_w = torch.split(attn.qkv_proj.weight, [nH * D, nKV * D, nKV * D], dim=0)
        if li == 0 and os.environ.get("ATOM_REFERENCE_WDUMP"):
            _wd = os.environ["ATOM_REFERENCE_WDUMP"]
            torch.save(
                {
                    "q_row0": q_w[0].detach().float().cpu(),
                    "k_row0": k_w[0].detach().float().cpu(),
                    "v_row0": v_w[0].detach().float().cpu(),
                    "fused_shape": torch.tensor(attn.qkv_proj.weight.shape),
                    "fused_contig": torch.tensor(int(attn.qkv_proj.weight.is_contiguous())),
                    "q_contig": torch.tensor(int(q_w.is_contiguous())),
                },
                _wd,
            )"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("weight dump added")
