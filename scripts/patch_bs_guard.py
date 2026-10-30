#!/usr/bin/env python3
"""Recompute mode holds a single token history, so it must refuse a multi-sequence
batch instead of silently mixing two sequences in one history."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """    bs = len(cu_q) - 1
    ctx_lens = md.context_lens.tolist()[:bs]"""
new = """    bs = len(cu_q) - 1
    if bs > 1 and os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        # the recompute history is single-sequence by construction
        print(
            f"[ref] refusing recompute with bs={bs} (single-sequence mode); "
            "set ATOM_REFERENCE_RECOMPUTE=0 for batched runs",
            flush=True,
        )
        return None
    ctx_lens = md.context_lens.tolist()[:bs]"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("multi-sequence guard added")
