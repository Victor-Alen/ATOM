#!/usr/bin/env python3
"""Save HF's per-layer last-row hidden to one file for offline comparisons."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/hf_layer_cmp.py")
s = p.read_text()
old = """for li in range(len(hs) - 1):"""
new = """torch.save([h[0, -1].float().cpu() for h in hs], "/workspace/logs/hf_hs.pt")
print("saved hf_hs.pt", flush=True)
for li in range(len(hs) - 1):"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("hf layer cache added")
