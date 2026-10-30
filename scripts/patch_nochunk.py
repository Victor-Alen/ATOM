#!/usr/bin/env python3
"""Exactness requires a single-shot attention: chunking the queries changes the
GEMM tiling and therefore the result (measured 0.875 at 4096 tokens). Make the
chunk size default to the whole sequence, with an explicit env opt-out."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = "            CH = 512"
new = ('            _ch = int(os.environ.get("ATOM_REFERENCE_ATTN_CHUNK", "0"))\n'
       "            CH = n_new if _ch <= 0 else _ch")
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("attention is single-shot by default")
