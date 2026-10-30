#!/usr/bin/env python3
"""Compute the softmax row-blockwise: identical per-row result, but the fp32 temp
drops from 7.9 GB to ~1 GB at 7881 tokens. The matmuls stay single-shot so the
GEMM tiling (and hence the bits) is unchanged."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
old = """                sc = torch.matmul(q[:, :, c0:c1], kk.transpose(2, 3)) * scaling
                sc = sc.masked_fill(~maskc.view(1, 1, c1 - c0, T), float("-inf"))
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, :, c0:c1] = torch.matmul(pr, vv)
                del sc, pr"""
new = """                sc = torch.matmul(q[:, :, c0:c1], kk.transpose(2, 3)) * scaling
                sc.masked_fill_(~maskc.view(1, 1, c1 - c0, T), float("-inf"))
                rows = sc.reshape(-1, T)
                pr = torch.empty_like(sc)
                prow = pr.reshape(-1, T)
                RB = 1024
                for r0 in range(0, rows.shape[0], RB):
                    r1 = min(r0 + RB, rows.shape[0])
                    prow[r0:r1] = F.softmax(rows[r0:r1], dim=-1, dtype=torch.float32)
                del rows, prow
                o_all[:, :, c0:c1] = torch.matmul(pr, vv)
                del sc, pr"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("probe softmax is row-blockwise now")
