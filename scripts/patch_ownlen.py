#!/usr/bin/env python3
"""Own-cache decode must attend over 0..max(position), i.e. include the new token;
md.context_lens excludes it, which silently dropped the current key."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = "    ctx_slots = seq_slots(positions, cu_q, ctx_lens, bs_bt, block_size, _recomp or _own, dev)"
new = """    if _own:
        ctx_lens = [
            int(positions[cu_q[_s]:cu_q[_s + 1]].max().item()) + 1 for _s in range(bs)
        ]
    ctx_slots = seq_slots(positions, cu_q, ctx_lens, bs_bt, block_size, _recomp or _own, dev)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("own-cache context length fixed")
