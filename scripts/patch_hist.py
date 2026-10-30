#!/usr/bin/env python3
"""Fix token-history bookkeeping: prefill replaces, decode appends."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if len(hist) >= len(new_ids):
            hist[:] = new_ids
        else:
            hist.extend(new_ids[len(hist):])"""
new = """        if len(new_ids) > 1:
            hist[:] = new_ids
        else:
            hist.append(new_ids[0])"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("history bookkeeping fixed")
