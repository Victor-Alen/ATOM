#!/usr/bin/env python3
"""The embedding is computed before the trim, so the trimmed ids never reached x.
Re-embed after the prologue adjustments."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if _fp == 0:
            _PENDING_FIRST[0] = True"""
new = """        if _fp == 0:
            _PENDING_FIRST[0] = True
        if int(x.shape[0]) != int(input_ids.numel()):
            x = F.embedding(input_ids, model.model.embed_tokens.weight)
            if x.dim() == 1:
                x = x[None, :]"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("re-embed after trim")
