#!/usr/bin/env python3
"""ATOM's sampler expects one logits row per request, while the reference path
returns one row per new token. Reduce to the last row of each request here."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = """            _logits = _F.linear(hidden_states, self.lm_head.weight)"""
new = """            _hs = hidden_states
            try:
                from atom.utils.forward_context import get_forward_context as _gfc

                _md = _gfc().attn_metadata
                if isinstance(_md, dict):
                    _md = next(iter(_md.values()))
                _cu = _md.cu_seqlens_q.tolist()
                if len(_cu) > 2 and _cu[-1] == _hs.shape[0]:
                    _idx = [_cu[i + 1] - 1 for i in range(len(_cu) - 1)]
                    _hs = _hs[_idx]
            except Exception:  # noqa: BLE001
                _hs = hidden_states[-1:]
            if _hs.shape[0] != 1 and len(getattr(_hs, "shape", ())) > 0:
                pass
            _logits = _F.linear(_hs, self.lm_head.weight)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("reference logits reduced to last row per request")
