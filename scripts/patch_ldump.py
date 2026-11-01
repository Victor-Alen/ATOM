#!/usr/bin/env python3
"""Latched per-layer dump: record every layer's attention and block output for the
first prefill of the process, in one file, so no comparison mixes two calls."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn"""
new = """    _LDUMP = {}
    _dump_dir = os.environ.get("ATOM_REFERENCE_DUMPL")
    _dump_call = bool(_dump_dir) and "layers" not in _DUMPED and int(n_new_total) > 1
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn"""
assert old in s
s = s.replace(old, new, 1)

old2 = """        if _ldir and n_new > 1:"""
new2 = """        if _dump_call:
            _LDUMP[f"attn_{li}"] = x[-1].detach().float().cpu()
        if _ldir and n_new > 1:"""
assert old2 in s
s = s.replace(old2, new2, 1)

old3 = """        if _ldir and n_new > 1:
            torch.save(x[-1].detach().float().cpu(), f"{_ldir}/blk_L{li}.pt")"""
new3 = """        if _dump_call:
            _LDUMP[f"blk_{li}"] = x[-1].detach().float().cpu()
        if _ldir and n_new > 1:
            torch.save(x[-1].detach().float().cpu(), f"{_ldir}/blk_L{li}.pt")"""
assert old3 in s
s = s.replace(old3, new3, 1)
p.write_text(s)
ast.parse(s)
print("per-layer latched dump added (needs n_new_total + save at the end)")
