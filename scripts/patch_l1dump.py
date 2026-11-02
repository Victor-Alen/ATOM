#!/usr/bin/env python3
"""Dump the full post-LN input and the q/k/v outputs of layer 1 so each sub-op can be
recomputed with torch from HF's weights."""
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
            if li == 1:
                _LDUMP["h1_full"] = h.detach().to(torch.bfloat16).cpu()
                _LDUMP["pos1"] = positions.detach().cpu()"""
assert old in s
s = s.replace(old, new, 1)

old2 = """            _lay("q_lin", q_lin)"""
new2 = """            if _dump_call and li == 1:
                _LDUMP["q1_full"] = q.detach().to(torch.bfloat16).cpu()
                _LDUMP["k1_full"] = k.detach().to(torch.bfloat16).cpu()
                _LDUMP["v1_full"] = v.detach().to(torch.bfloat16).cpu()
            _lay("q_lin", q_lin)"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("layer-1 sub-op tensors added to the dump")
