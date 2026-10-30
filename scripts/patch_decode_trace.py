#!/usr/bin/env python3
"""Trace the decode step: ids, embedding, per-layer and final values."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    x = F.embedding(input_ids, model.model.embed_tokens.weight)
    if x.dim() == 1:
        x = x[None, :]"""
new = """    x = F.embedding(input_ids, model.model.embed_tokens.weight)
    if x.dim() == 1:
        x = x[None, :]
    if os.environ.get("ATOM_REFERENCE_DEBUG") in ("0", "1", "2"):
        print(
            f"[ref] ids={input_ids[:4].tolist()} n={input_ids.numel()} "
            f"emb_absmax={float(x.abs().max()):.4f}",
            flush=True,
        )"""
assert old in s
s = s.replace(old, new, 1)

old2 = """    return _hf_norm(x, model.model.norm.weight, eps)"""
new2 = """    if os.environ.get("ATOM_REFERENCE_DEBUG") in ("0", "1", "2"):
        print(f"[ref] final_absmax={float(x.abs().max()):.4f}", flush=True)
    return _hf_norm(x, model.model.norm.weight, eps)"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("decode trace added")
