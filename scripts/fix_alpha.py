#!/usr/bin/env python3
"""Fix the undefined _alpha: compute it inline at the call site."""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

bad = "        swiglu_alpha=_alpha," + NL
good = (
    "        swiglu_alpha=float(__import__('os').environ.get('ATOM_TRITON_ALPHA') or 1.702)," + NL
)
n = src.count(bad)
src = src.replace(bad, good)
# remove now-unused local alpha definition if present
stale = (
    "    _alpha_env = __import__('os').environ.get('ATOM_TRITON_ALPHA')" + NL
    + "    _alpha = float(_alpha_env) if _alpha_env else 1.702" + NL
)
m = src.count(stale)
src = src.replace(stale, "")

F.write_text(src)
ast.parse(src)
print(f"replaced {n} call sites, removed {m} stale blocks")
