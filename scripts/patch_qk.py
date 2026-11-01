#!/usr/bin/env python3
"""Dump post-rope q/k for the last position of layer 0 (the first divergence sits
there), so it can be compared against HF in-engine."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin"""
new = """            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            _ld3 = os.environ.get("ATOM_REFERENCE_DUMPL")
            if _ld3 and li == 0 and n_new is None:
                pass
            if _ld3 and li == 0 and q.shape[0] > 1:
                torch.save(q[-1].detach().float().cpu(), f"{_ld3}/q_L0.pt")
                torch.save(k[-1].detach().float().cpu(), f"{_ld3}/k_L0.pt")
                torch.save(h[-1].detach().float().cpu(), f"{_ld3}/h_ln_L0.pt")"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("q/k/h_ln dumps added at L0")
