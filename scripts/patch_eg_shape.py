#!/usr/bin/env python3
"""x must be 2-D: embedding of a [1, T] ids tensor is [1, T, H]."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/exact_generate.py")
s = p.read_text()
old = "            x = F.embedding(ids, embed)"
new = "            x = F.embedding(ids, embed)[0]"
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
import ast
ast.parse(s)
print("x squeezed to 2-D")
