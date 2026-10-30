#!/usr/bin/env python3
"""Add cache-key diagnostics for the decode path."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    keys = [int(bt[s, 0].item()) for s in range(bs)]"""
new = """    keys = [int(bt[s, 0].item()) for s in range(bs)]
    if os.environ.get("ATOM_REFERENCE_DEBUG") in ("1", "2"):
        print(f"[ref] bs={bs} cu_q={cu_q} ctx={md.context_lens.tolist()[:bs]} keys={keys}", flush=True)"""
assert old in s
s = s.replace(old, new, 1)

old2 = """            ck = _CACHE.get((li, keys[s]))"""
new2 = """            ck = _CACHE.get((li, keys[s]))
            if os.environ.get("ATOM_REFERENCE_DEBUG") == "2" and li == 0:
                print(f"[ref] li={li} key={keys[s]} q_len={hi - lo} cached={0 if ck is None else ck[0].shape[1]}", flush=True)"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("diagnostics added")
