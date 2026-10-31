#!/usr/bin/env python3
"""Add an optional per-KV-head attention loop (ATOM_REFERENCE_HEAD_LOOP=1).

Each slice still has shape [rep*T, T], the same shape as one slice of the batched
matmul, so the arithmetic per element is unchanged; the point is to keep every
temporary below the int32 element count, which is what breaks at ~12.5k tokens.
"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """                qc = qh[:, :, c0:c1, :].reshape(nKV, rep * (c1 - c0), D)
                sc = torch.matmul(qc, kk.transpose(1, 2)) * scale
                neg = ~mask.repeat(rep, 1).view(1, (c1 - c0) * rep, ctx)
                sc.masked_fill_(neg, float("-inf"))
                del neg"""
new = """                _hl = os.environ.get("ATOM_REFERENCE_HEAD_LOOP") == "1"
                if _hl:
                    neg2 = ~mask.repeat(rep, 1)
                    for _hk in range(nKV):
                        _qc = qh[_hk, :, c0:c1, :].reshape(rep * (c1 - c0), D)
                        _sc = torch.matmul(_qc, kk[_hk].transpose(0, 1)) * scale
                        _sc.masked_fill_(neg2, float("-inf"))
                        _pr32 = F.softmax(_sc, dim=-1, dtype=torch.float32)
                        del _sc
                        _prb = _pr32.to(q.dtype)
                        del _pr32
                        o_all[_hk, :, c0:c1, :] = torch.matmul(_prb, vv[_hk]).view(
                            rep, c1 - c0, D
                        )
                        del _prb, _qc
                    del neg2
                    continue
                qc = qh[:, :, c0:c1, :].reshape(nKV, rep * (c1 - c0), D)
                sc = torch.matmul(qc, kk.transpose(1, 2)) * scale
                neg = ~mask.repeat(rep, 1).view(1, (c1 - c0) * rep, ctx)
                sc.masked_fill_(neg, float("-inf"))
                del neg"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("per-KV-head attention loop added (env-gated)")
