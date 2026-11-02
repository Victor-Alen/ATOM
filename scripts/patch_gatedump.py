#!/usr/bin/env python3
"""Dump the layer-0 gate top-8 selection once, to compare against HF at the positions
where the outputs differ."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    token_idx = torch.arange(tokens, device=hs.device).unsqueeze(1).expand(-1, top_k).reshape(-1)"""
new = """    _gd = os.environ.get("ATOM_REFERENCE_GATEDUMP")
    if _gd and "gate" not in _DUMPED:
        _DUMPED.add("gate")
        torch.save(
            {
                "gate_logits": logits_r.detach().float().cpu(),
                "topi": topi.detach().cpu(),
                "topv": topv.detach().float().cpu(),
            },
            _gd,
        )
    token_idx = torch.arange(tokens, device=hs.device).unsqueeze(1).expand(-1, top_k).reshape(-1)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("gate dump added")
