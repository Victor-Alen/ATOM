#!/usr/bin/env python3
"""A new prefill can arrive with a trailing deferred token from the previous request
(observed: prompt of 1470 tokens delivered as 1471 with the old sample appended).
The engine's ctx_lens carries the real token count, so truncate to it."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        new_ids = input_ids.tolist()
        _fp = int(positions[0].item()) if positions.numel() else 0"""
new = """        _ctxs = None
        try:
            _ctxs = [int(v) for v in md.context_lens.tolist()]
        except Exception:  # noqa: BLE001
            _ctxs = None
        new_ids = input_ids.tolist()
        _fp = int(positions[0].item()) if positions.numel() else 0
        if _fp == 0 and _ctxs and _ctxs[0] == len(new_ids) - 1:
            new_ids = new_ids[:-1]
            positions = positions[:-1]"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("trailing deferred token dropped for a new prefill")
