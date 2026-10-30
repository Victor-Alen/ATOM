#!/usr/bin/env python3
"""Broadcast rope tables over heads: cos/sin [T, D] -> [T, 1, D]."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = "            cos, sin = rot(li, positions[lo:hi], x.dtype)"
new = ("            cos, sin = rot(li, positions[lo:hi], x.dtype)\n"
       "            cos = cos.unsqueeze(1)\n"
       "            sin = sin.unsqueeze(1)")
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("rope broadcast fixed")
