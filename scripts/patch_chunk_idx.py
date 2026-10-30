#!/usr/bin/env python3
"""Fix the query-chunk slicing: rows are (head, token) with token fastest, so a
time chunk must be taken after reshaping to [nKV, rep, T, D]."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = s[s.index("            qh = q.transpose(0, 1).reshape(nKV, rep * n_new, D)"):s.index("            outs[lo:hi] = F.linear(o, attn.o_proj.weight)")]
new = '''            qh = q.transpose(0, 1).view(nKV, rep, n_new, D)
            ki = torch.arange(ctx, device=dev).view(1, -1)
            CH = 512
            o_all = torch.empty(nKV, rep, n_new, D, dtype=q.dtype, device=dev)
            for c0 in range(0, n_new, CH):
                c1 = min(c0 + CH, n_new)
                qi = new_pos[c0:c1].view(-1, 1)
                mask = ki <= qi
                if lt == "sliding_attention" and W is not None:
                    mask = mask & ((qi - ki) < W)
                qc = qh[:, :, c0:c1, :].reshape(nKV, rep * (c1 - c0), D)
                sc = torch.matmul(qc, kk.transpose(1, 2)) * scale
                sc = sc.masked_fill(
                    ~mask.repeat(rep, 1).view(1, (c1 - c0) * rep, ctx), float("-inf")
                )
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, :, c0:c1, :] = torch.matmul(pr, vv).view(nKV, rep, c1 - c0, D)
                del sc, pr, qc
            o = o_all.permute(2, 0, 1, 3).reshape(n_new, nH * D)
'''
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("query chunk slicing fixed")
