#!/usr/bin/env python3
"""Reduce the transient memory of the exact single-shot attention without
changing its arithmetic: mask in place, free the scores before the cast."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """                sc = torch.matmul(qc, kk.transpose(1, 2)) * scale
                sc = sc.masked_fill(
                    ~mask.repeat(rep, 1).view(1, (c1 - c0) * rep, ctx), float("-inf")
                )
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, :, c0:c1, :] = torch.matmul(pr, vv).view(nKV, rep, c1 - c0, D)
                del sc, pr, qc"""
new = """                sc = torch.matmul(qc, kk.transpose(1, 2)) * scale
                neg = ~mask.repeat(rep, 1).view(1, (c1 - c0) * rep, ctx)
                sc.masked_fill_(neg, float("-inf"))
                del neg
                pr32 = F.softmax(sc, dim=-1, dtype=torch.float32)
                del sc
                pr = pr32.to(q.dtype)
                del pr32
                o_all[:, :, c0:c1, :] = torch.matmul(pr, vv).view(nKV, rep, c1 - c0, D)
                del pr, qc"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("attention peak memory reduced in place")
