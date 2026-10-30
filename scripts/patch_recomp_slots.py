#!/usr/bin/env python3
"""Recompute mode must not depend on the engine's block table: keep our own
sequential KV slots for 0..len(hist)-1 and grow the buffer when needed."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

s = s.replace("""    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        key = int(bs_bt[0, 0].item())""",
"""    _recomp = False
    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        _recomp = True
        key = int(bs_bt[0, 0].item())""", 1)

s = s.replace("""    ctx_slots = []
    for s in range(bs):
        p = positions[cu_q[s]:cu_q[s + 1]].long()""",
"""    ctx_slots = []
    for s in range(bs):
        if _recomp:
            ctx_slots.append(torch.arange(int(ctx_lens[s]), device=dev))
            continue
        p = positions[cu_q[s]:cu_q[s + 1]].long()""", 1)

s = s.replace("""        if li not in _KV:
            max_slot = int(bs_bt.max().item()) * block_size + block_size""",
"""        _need = int(bs_bt.max().item()) * block_size + block_size
        if _recomp:
            _need = max(_need, int(ctx_lens[0]) + 1)
        if li not in _KV or _KV[li][0].shape[0] < _need:
            max_slot = _need""", 1)

p.write_text(s)
ast.parse(s)
print("recompute uses its own sequential slots")
