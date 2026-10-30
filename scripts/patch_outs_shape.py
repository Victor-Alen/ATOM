#!/usr/bin/env python3
"""outs must be [T, hidden] (o_proj output), not [T, nH*D]."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = "        outs = torch.empty(x.shape[0], nH * D, dtype=x.dtype, device=dev)"
new = "        outs = torch.empty(x.shape[0], cfg.hidden_size, dtype=x.dtype, device=dev)"
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
import ast
ast.parse(s)
print("outs shape fixed")
