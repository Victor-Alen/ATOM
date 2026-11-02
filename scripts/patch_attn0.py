#!/usr/bin/env python3
"""Dump layer-0's full attention output for a position-wise comparison with HF."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if _dump_call and li < 4:
            _LDUMP[f"h_{li}"] = h[-1].detach().float().cpu()
            _LDUMP[f"attn_out_{li}"] = outs[-1].detach().float().cpu()"""
new = """        if _dump_call and li < 4:
            _LDUMP[f"h_{li}"] = h[-1].detach().float().cpu()
            _LDUMP[f"attn_out_{li}"] = outs[-1].detach().float().cpu()
            if li == 0:
                _LDUMP["attn0_full"] = outs.detach().to(torch.bfloat16).cpu()"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("attn0 full dump added")
