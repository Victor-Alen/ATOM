#!/usr/bin/env python3
"""Fix the rotary call site (layer index, not a dummy tensor)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """            cos, sin = rot(torch.zeros(1, hi - lo, cfg.hidden_size, dtype=x.dtype, device=dev),
                           positions[lo:hi].view(1, -1), lt)"""
new = """            cos, sin = rot(li, positions[lo:hi], x.dtype)"""
assert old in s, "call site not found"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("rotary call site fixed")
