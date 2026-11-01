#!/usr/bin/env python3
"""Dump the engine's last-row hidden state so it can be compared with HF directly."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = """            _logits = _F.linear(_hs, self.lm_head.weight)"""
new = """            _dmp = None
            try:
                import os as _os3

                _dmp = _os3.environ.get("ATOM_REFERENCE_DUMPH")
            except Exception:  # noqa: BLE001
                _dmp = None
            if _dmp:
                try:
                    torch.save(_hs[-1].detach().float().cpu(), _dmp)
                except Exception:  # noqa: BLE001
                    pass
            _logits = _F.linear(_hs, self.lm_head.weight)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("hidden dump added")
