#!/usr/bin/env python3
"""Own-cache must not reallocate the KV buffer when the sequence grows: the growth
check wiped the prefill cache, so decode attended over zeros. Size it for the whole
context up front instead."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if _recomp or _own:
            _need = int(ctx_lens[0]) + 1 if _recomp else int(positions.max().item()) + 1"""
new = """        if _own:
            _need = int(getattr(cfg, "max_position_embeddings", 32768)) + 1
        elif _recomp:
            _need = int(ctx_lens[0]) + 1"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("own-cache KV sized for the whole context")
