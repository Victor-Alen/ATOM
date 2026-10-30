#!/usr/bin/env python3
"""Probe: make the query chunk optional (0 = single shot, exact)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
if "PROBE_CHUNK" not in s:
    s = s.replace("            CH = 512",
                  '            CH = int(__import__("os").environ.get("PROBE_CHUNK", "0")) or T', 1)
    p.write_text(s)
    ast.parse(s)
    print("probe chunk now optional")
else:
    print("already patched")
