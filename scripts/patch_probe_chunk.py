#!/usr/bin/env python3
"""Chunk the probe's attention queries so a 7881-token forward fits in memory.

This mirrors the chunked path used inside ATOM, so the result also validates it.
"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
old = s[s.index("            kk = k.repeat_interleave(nH // nKV, dim=1)"):s.index("            o = torch.matmul(probs, vv).transpose(1, 2).contiguous().view(1, T, nH * D)")]
new = '''            rep = nH // nKV
            kk = k.repeat_interleave(rep, dim=1)
            vv = v.repeat_interleave(rep, dim=1)
            ki = torch.arange(T, device=dev).view(1, -1)
            qi_all = pos
            CH = 512
            o_all = torch.empty(nH, T, D, dtype=q.dtype, device=dev)
            for c0 in range(0, T, CH):
                c1 = min(c0 + CH, T)
                qi = qi_all[c0:c1].view(-1, 1)
                maskc = ki <= qi
                if attn.sliding_window is not None and args.window_rule != "none":
                    if args.window_rule == "lt":
                        maskc = maskc & ((qi - ki) < W)
                    else:
                        maskc = maskc & ((qi - ki) <= W)
                sc = torch.matmul(q[:, :, c0:c1], kk.transpose(1, 2)) * scaling
                sc = sc.masked_fill(~maskc.view(1, c1 - c0, T), float("-inf"))
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, c0:c1] = torch.matmul(pr, vv)
                del sc, pr
            o = o_all.transpose(1, 2).contiguous().view(1, T, nH * D)
'''
s = s.replace(old, new, 1)
# drop the now-unused lines that followed
s = s.replace("            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)\n", "")
s = s.replace("            del scores, probs, o, kk, vv\n", "            del o, kk, vv, o_all\n")
p.write_text(s)
ast.parse(s)
print("probe attention chunked")
