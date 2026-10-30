#!/usr/bin/env python3
from pathlib import Path
p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
s = s.replace("del routed, final, h, h2, moe_out", "del routed, h, h2, moe_out")
p.write_text(s)
import ast
ast.parse(s)
print("stale 'final' reference removed")
