#!/usr/bin/env python3
"""After trimming, cu_seqlens_q still advertised the old token count, so the layer loop
sliced a 3895-row x with a 3896-token range. Rebuild the per-request geometry."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if int(x.shape[0]) != int(input_ids.numel()):
            x = F.embedding(input_ids, model.model.embed_tokens.weight)
            if x.dim() == 1:
                x = x[None, :]"""
new = """        cu_q = [0, len(new_ids)]
        ctx_lens = [len(new_ids)]
        bs = 1
        if int(x.shape[0]) != int(input_ids.numel()):
            x = F.embedding(input_ids, model.model.embed_tokens.weight)
            if x.dim() == 1:
                x = x[None, :]"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("geometry rebuilt after trim")
