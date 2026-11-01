#!/usr/bin/env python3
"""The h_ln dump was overwritten by decode calls; keep only the prefill."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """                torch.save(h[-1].detach().float().cpu(), f"{_ld3}/h_ln_L0.pt")"""
new = """                if hi - lo > 1:
                    torch.save(h[-1].detach().float().cpu(), f"{_ld3}/h_ln_L0.pt")"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("h_ln dump restricted to the prefill")
