#!/usr/bin/env python3
"""Recompute mode does not need the engine's block table at all: allow None there
(this is exactly what failed on the decode step and forced a fallback)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
s = s.replace("""    bs_bt = md.block_tables
    if bs_bt is None:
        return None""",
"""    bs_bt = md.block_tables
    if bs_bt is None and os.environ.get("ATOM_REFERENCE_RECOMPUTE") != "1":
        return None""", 1)
s = s.replace("        key = int(bs_bt[0, 0].item())",
              "        key = int(bs_bt[0, 0].item()) if bs_bt is not None else 0", 1)
s = s.replace("""        _need = int(bs_bt.max().item()) * block_size + block_size
        if _recomp:
            _need = max(_need, int(ctx_lens[0]) + 1)""",
"""        _need = int(bs_bt.max().item()) * block_size + block_size if bs_bt is not None else 0
        if _recomp:
            _need = max(_need, int(ctx_lens[0]) + 1)""", 1)
p.write_text(s)
ast.parse(s)
print("recompute tolerates a missing block table")
