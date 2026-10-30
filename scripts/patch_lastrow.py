#!/usr/bin/env python3
"""Print last-row absmax per layer (comparable with the HF golden fingerprints)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if os.environ.get("ATOM_REFERENCE_DEBUG") == "2" and li < 3:
            print(f"[ref] layer {li} absmax={float(x.abs().max()):.4f}", flush=True)"""
new = """        if os.environ.get("ATOM_REFERENCE_DEBUG") == "2" and li < 8:
            print(f"[ref] L{li} last_absmax={float(x[-1].abs().max()):.4f}", flush=True)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("last-row debug ready")
