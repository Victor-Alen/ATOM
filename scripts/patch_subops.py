#!/usr/bin/env python3
"""Record layer 0-3 sub-operations (post-LN, attention out, post-attn LN, MoE out) in the
latched dump so the one-ulp difference can be pinned to a single op."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """        if _dump_call:
            _LDUMP[f"attn_{li}"] = x[-1].detach().float().cpu()"""
new = """        if _dump_call and li < 4:
            _LDUMP[f"h_{li}"] = h[-1].detach().float().cpu()
            _LDUMP[f"attn_out_{li}"] = outs[-1].detach().float().cpu()
        if _dump_call:
            _LDUMP[f"attn_{li}"] = x[-1].detach().float().cpu()"""
assert old in s
s = s.replace(old, new, 1)

old2 = """        if _dump_call:
            _LDUMP[f"blk_{li}"] = x[-1].detach().float().cpu()"""
new2 = """        if _dump_call and li < 4:
            _LDUMP[f"h2_{li}"] = h2[-1].detach().float().cpu()
            _LDUMP[f"moe_{li}"] = moe[-1].detach().float().cpu()
        if _dump_call:
            _LDUMP[f"blk_{li}"] = x[-1].detach().float().cpu()"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("sub-op dumps added for layers 0-3")
