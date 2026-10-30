#!/usr/bin/env python3
"""reference_mode: fall back to the normal path when the batch has no block table
(warmup / graph capture), and let the caller handle it."""
from pathlib import Path

r = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = r.read_text()
old = """    cu_q = md.cu_seqlens_q.tolist()
    bs = len(cu_q) - 1
    ctx_lens = md.context_lens.tolist()[:bs]
    bt = md.block_tables
    keys = [int(bt[s, 0].item()) for s in range(bs)]"""
new = """    cu_q = md.cu_seqlens_q.tolist()
    bs = len(cu_q) - 1
    bt = md.block_tables
    if bt is None:
        return None
    keys = [int(bt[s, 0].item()) for s in range(bs)]"""
assert old in s
s = s.replace(old, new, 1)
r.write_text(s)

m = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
t = m.read_text()
old2 = """            from atom.models.reference_mode import reference_forward

            return reference_forward(self, input_ids, positions)"""
new2 = """            from atom.models.reference_mode import reference_forward

            _ref = reference_forward(self, input_ids, positions)
            if _ref is not None:
                return _ref"""
assert old2 in t
t = t.replace(old2, new2, 1)
m.write_text(t)
import ast
ast.parse(s)
ast.parse(t)
print("fallback wired")
