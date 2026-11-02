#!/usr/bin/env python3
"""Repair the indentation the previous patch mangled."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
bad = """            else:
                    _w13 = _unshuffle(exp.w13_weight)
                _w2 = _unshuffle(exp.w2_weight)
"""
good = """            else:
                _w13 = _unshuffle(exp.w13_weight)
                _w2 = _unshuffle(exp.w2_weight)
"""
assert bad in s, "pattern not found"
s = s.replace(bad, good, 1)
p.write_text(s)
ast.parse(s)
print("indentation repaired")
