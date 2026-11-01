#!/usr/bin/env python3
"""Record the first few positions and ids so the extra leading token can be identified."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        "emb_last": x[-1].detach().float().cpu() if int(x.shape[0]) > 1 else None,"""
new = """        "emb_last": x[-1].detach().float().cpu() if int(x.shape[0]) > 1 else None,
        "ids_head": input_ids[:4].detach().cpu().tolist(),
        "ids_tail": input_ids[-4:].detach().cpu().tolist(),
        "pos_head": positions[:4].detach().cpu().tolist() if positions.numel() else [],"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("head/tail recorded")
