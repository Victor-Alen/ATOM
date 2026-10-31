#!/usr/bin/env python3
"""In recompute mode the reference path owns the KV buffer, so size it to the
history length instead of the engine's whole block table (with util 0.45 the block
table implies ~80k slots per layer, i.e. ~9 GB of zero-fill for a 16-token prompt)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        _need = int(bs_bt.max().item()) * block_size + block_size if bs_bt is not None else 0
        if _recomp:
            _need = max(_need, int(ctx_lens[0]) + 1)"""
new = """        if _recomp:
            _need = int(ctx_lens[0]) + 1
        else:
            _need = int(bs_bt.max().item()) * block_size + block_size"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("KV buffer sized to the history in recompute mode")
