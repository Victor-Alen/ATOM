#!/usr/bin/env python3
"""Remove the stray guard that referenced n_new before assignment."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """            if _ld3 and li == 0 and n_new is None:
                pass
"""
assert old in s
s = s.replace(old, "", 1)
p.write_text(s)
ast.parse(s)
print("stray guard removed")
