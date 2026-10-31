#!/usr/bin/env python3
"""The recompute block rewrote input_ids AFTER the embedding was computed, so a
decode call had x of 1 row while the loop wrote 17 -> shape mismatch."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
anchor = """        cu_q = [0, len(hist)]
        bs = 1
        ctx_lens = [len(hist)]"""
new = """        cu_q = [0, len(hist)]
        bs = 1
        ctx_lens = [len(hist)]
        x = F.embedding(input_ids, model.model.embed_tokens.weight)
        if x.dim() == 1:
            x = x[None, :]"""
assert anchor in s
s = s.replace(anchor, new, 1)
p.write_text(s)
ast.parse(s)
print("re-embed after the recompute history is applied")
