#!/usr/bin/env python3
"""The engine hands over a padded input_ids buffer, so a shorter new request inherits
stale tail tokens from the previous one (observed: 1470-token prompt arriving as 1471
with the previous sample appended). Truncate to the batch token count from cu_seqlens_q."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        _ctxs = None
        try:
            _ctxs = [int(v) for v in md.context_lens.tolist()]
        except Exception:  # noqa: BLE001
            _ctxs = None
        new_ids = input_ids.tolist()"""
new = """        try:
            _ntok = int(md.cu_seqlens_q[-1])
        except Exception:  # noqa: BLE001
            _ntok = int(input_ids.numel())
        if 0 < _ntok < int(input_ids.numel()):
            input_ids = input_ids[:_ntok]
            if positions.numel() > _ntok:
                positions = positions[:_ntok]
        new_ids = input_ids.tolist()"""
assert old in s
s = s.replace(old, new, 1)
old2 = """        if _fp == 0 and _ctxs and _ctxs[0] == len(new_ids) - 1:
            new_ids = new_ids[:-1]
            positions = positions[:-1]"""
assert old2 in s
s = s.replace(old2, "", 1)
p.write_text(s)
ast.parse(s)
print("input_ids truncated to cu_seqlens_q[-1]")
