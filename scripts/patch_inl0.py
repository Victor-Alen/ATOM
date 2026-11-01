#!/usr/bin/env python3
"""Dump the layer-0 input row as well, so embedding vs attention can be told apart."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        x = x + outs
        _ldir = os.environ.get("ATOM_REFERENCE_DUMPL")"""
new = """        x = x + outs
        _ldir = os.environ.get("ATOM_REFERENCE_DUMPL")
        if _ldir and n_new > 1 and li == 0:
            torch.save((x[-1] - outs[-1]).detach().float().cpu(), f"{_ldir}/in_L0.pt")"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("layer-0 input dump added")
