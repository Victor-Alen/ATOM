#!/usr/bin/env python3
"""Finish the latched per-layer dump: correct the size probe and persist the file."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
s = s.replace(
    '    _dump_call = bool(_dump_dir) and "layers" not in _DUMPED and int(n_new_total) > 1',
    '    _dump_call = bool(_dump_dir) and "layers" not in _DUMPED and int(x.shape[0]) > 1',
    1,
)
old = """    out = _hf_norm(x, model.model.norm.weight, eps)
    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        return out[keep]
    return out"""
new = """    out = _hf_norm(x, model.model.norm.weight, eps)
    if _dump_call:
        _DUMPED.add("layers")
        _LDUMP["final"] = out[-1].detach().float().cpu()
        torch.save(_LDUMP, f"{_dump_dir}/layers.pt")
    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        return out[keep]
    return out"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("latched layer dump finalised")
