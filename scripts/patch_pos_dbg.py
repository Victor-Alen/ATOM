#!/usr/bin/env python3
"""Print positions on every call to confirm their semantics during decode."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    rot, layer_types, W = _rotary(cfg, dev)"""
new = """    rot, layer_types, W = _rotary(cfg, dev)
    if os.environ.get("ATOM_REFERENCE_DEBUG") in ("0", "1", "2"):
        print(
            f"[ref] pos={positions.tolist()[:4]}..{positions.tolist()[-2:]} "
            f"n={positions.numel()} cu_q={cu_q[:3]} ctx={ctx_lens[:3]}",
            flush=True,
        )"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("position debug added")
