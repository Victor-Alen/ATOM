#!/usr/bin/env python3
"""Add ATOM_REFERENCE_OWN_CACHE=1: normal incremental decoding (no recompute) while
the reference path owns the KV buffer at sequential slots. Decode steps arrive with
block_tables=None, so this is what makes a 64-step scenario run cheaply: each step
only processes the new token instead of re-running the whole prefix."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """    bs_bt = md.block_tables
    if bs_bt is None and os.environ.get("ATOM_REFERENCE_RECOMPUTE") != "1":
        return None"""
new = """    _own = os.environ.get("ATOM_REFERENCE_OWN_CACHE") == "1"
    bs_bt = md.block_tables
    if bs_bt is None and os.environ.get("ATOM_REFERENCE_RECOMPUTE") != "1" and not _own:
        return None"""
assert old in s
s = s.replace(old, new, 1)

old2 = "    ctx_slots = seq_slots(positions, cu_q, ctx_lens, bs_bt, block_size, _recomp, dev)"
new2 = "    ctx_slots = seq_slots(positions, cu_q, ctx_lens, bs_bt, block_size, _recomp or _own, dev)"
assert old2 in s
s = s.replace(old2, new2, 1)

old3 = """            if _recomp:
                write_slots = new_pos
            else:"""
new3 = """            if _recomp or _own:
                write_slots = new_pos
            else:"""
assert old3 in s
s = s.replace(old3, new3, 1)

old4 = """        if _recomp:
            _need = int(ctx_lens[0]) + 1
        else:
            _need = int(bs_bt.max().item()) * block_size + block_size"""
new4 = """        if _recomp or _own:
            _need = int(ctx_lens[0]) + 1 if _recomp else int(positions.max().item()) + 1
        else:
            _need = int(bs_bt.max().item()) * block_size + block_size"""
assert old4 in s
s = s.replace(old4, new4, 1)
p.write_text(s)
ast.parse(s)
print("own-cache incremental mode added")
