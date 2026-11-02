#!/usr/bin/env python3
"""Dump the raw vs unshuffled expert weights for layer 0 so the layout hypothesis can
be settled against the checkpoint."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """            _w13 = _unshuffle(exp.w13_weight)
                _w2 = _unshuffle(exp.w2_weight)"""
if old not in s:
    old = """                _w13 = _unshuffle(exp.w13_weight)
                _w2 = _unshuffle(exp.w2_weight)"""
new = """                _w13 = _unshuffle(exp.w13_weight)
                _w2 = _unshuffle(exp.w2_weight)
                _wd2 = os.environ.get("ATOM_REFERENCE_WDUMP2")
                if _wd2 and "w" not in _DUMPED:
                    _DUMPED.add("w")
                    torch.save(
                        {
                            "raw_w13_e0": exp.w13_weight[0].detach().to(torch.bfloat16).cpu(),
                            "raw_w2_e0": exp.w2_weight[0].detach().to(torch.bfloat16).cpu(),
                            "un_w13_e0": _w13[0].detach().to(torch.bfloat16).cpu(),
                            "raw_shape": torch.tensor(exp.w13_weight.shape),
                        },
                        _wd2,
                    )"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("raw weight dump added")
