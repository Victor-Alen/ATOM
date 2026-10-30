#!/usr/bin/env python3
"""Memory-lean attention: group heads by KV head instead of materialising
repeat_interleave copies, and shrink the query chunk."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = s[s.index("            sl = ctx_slots[s]"):s.index("            o = o_all.reshape(-1, nH * D)")]
new = '''            sl = ctx_slots[s]
            kk = k_flat[sl].transpose(0, 1)  # [nKV, ctx, D]
            vv = v_flat[sl].transpose(0, 1)
            rep = nH // nKV
            qh = q.transpose(0, 1).reshape(nKV, rep * n_new, D)
            ki = torch.arange(ctx, device=dev).view(1, -1)
            CH = 512
            o_all = torch.empty(nKV, rep * n_new, D, dtype=q.dtype, device=dev)
            for c0 in range(0, n_new, CH):
                c1 = min(c0 + CH, n_new)
                qi = new_pos[c0:c1].view(-1, 1)
                mask = ki <= qi
                if lt == "sliding_attention" and W is not None:
                    mask = mask & ((qi - ki) < W)
                qc = qh[:, c0 * rep:(c1) * rep, :]
                sc = torch.matmul(qc, kk.transpose(1, 2)) * scale
                sc = sc.masked_fill(~mask.repeat(rep, 1).view(1, (c1 - c0) * rep, ctx), float("-inf"))
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, c0 * rep:c1 * rep, :] = torch.matmul(pr, vv)
                del sc, pr, qc
            o = (
                o_all.view(nKV, rep, n_new, D)
                .permute(2, 0, 1, 3)
                .reshape(n_new, nH * D)
            )
'''
s = s.replace(old, new, 1)
s = s.replace("            del o, o_all, krep, vrep, kk, vv\n", "            del o, o_all, kk, vv\n")
p.write_text(s)
ast.parse(s)
print("attention rewritten memory-lean")
