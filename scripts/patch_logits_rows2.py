#!/usr/bin/env python3
"""The reduction condition required more than one request, so single-request
prefill (cu_seqlens_q=[0,16]) was left with 16 logits rows."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = """                _cu = _md.cu_seqlens_q.tolist()
                if len(_cu) > 2 and _cu[-1] == _hs.shape[0]:
                    _idx = [_cu[i + 1] - 1 for i in range(len(_cu) - 1)]
                    _hs = _hs[_idx]
            except Exception:  # noqa: BLE001
                _hs = hidden_states[-1:]"""
new = """                _cu = _md.cu_seqlens_q.tolist()
                _n_req = len(_cu) - 1
                if _n_req >= 1 and int(_cu[-1]) == _hs.shape[0]:
                    _idx = [_cu[i + 1] - 1 for i in range(_n_req)]
                    _hs = _hs[_idx]
                else:
                    _hs = hidden_states[-1:]
            except Exception:  # noqa: BLE001
                _hs = hidden_states[-1:]"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("reduction now applies to a single request too")
