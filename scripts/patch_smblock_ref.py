#!/usr/bin/env python3
"""Optional row-blocked softmax (ATOM_REFERENCE_SM_BLOCK=N, 0 disables).

Each row is normalised independently, so blocking the rows changes memory only, not
arithmetic. It removes the 32 x T x T fp32 temporary that makes a 12.5k-token prompt
need ~20 GB at once.
"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """                pr32 = F.softmax(sc, dim=-1, dtype=torch.float32)
                del sc
                pr = pr32.to(q.dtype)
                del pr32"""
new = """                _smb = int(os.environ.get("ATOM_REFERENCE_SM_BLOCK", "0"))
                if _smb > 0:
                    _rows = sc.reshape(-1, ctx)
                    pr = torch.empty_like(sc)
                    _prow = pr.reshape(-1, ctx)
                    for _r0 in range(0, _rows.shape[0], _smb):
                        _r1 = min(_r0 + _smb, _rows.shape[0])
                        _prow[_r0:_r1] = F.softmax(_rows[_r0:_r1], dim=-1, dtype=torch.float32)
                    del _rows, _prow
                    del sc
                else:
                    pr32 = F.softmax(sc, dim=-1, dtype=torch.float32)
                    del sc
                    pr = pr32.to(q.dtype)
                    del pr32"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("row-blocked softmax added (env-gated)")
