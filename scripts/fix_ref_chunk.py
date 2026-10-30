#!/usr/bin/env python3
"""reference_mode: chunk the query rows of the attention so a 16k prefill does not
materialise a 32 x 16384 x 16384 score tensor."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """            S = kk.shape[1]
            rep = nH // nKV
            krep = kk.repeat_interleave(rep, dim=0)
            vrep = vv.repeat_interleave(rep, dim=0)
            scores = torch.matmul(q, krep.transpose(1, 2)) * scale
            qi = positions[lo:hi].view(-1, 1)
            ki = torch.arange(S, device=dev).view(1, -1)
            mask = ki <= qi
            if lt == "sliding_attention" and W is not None:
                mask = mask & ((qi - ki) < W)
            scores = scores.masked_fill(~mask.view(1, hi - lo, S), float("-inf"))
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            o = torch.matmul(probs, vrep).transpose(0, 1).reshape(-1, nH * D)
            outs[lo:hi] = F.linear(o, attn.o_proj.weight)
            del scores, probs, o, krep, vrep"""
new = """            S = kk.shape[1]
            rep = nH // nKV
            krep = kk.repeat_interleave(rep, dim=0)
            vrep = vv.repeat_interleave(rep, dim=0)
            ki = torch.arange(S, device=dev).view(1, -1)
            qi_all = positions[lo:hi]
            CH = 2048
            o_all = torch.empty(hi - lo, nH, D, dtype=q.dtype, device=dev)
            for c0 in range(0, hi - lo, CH):
                c1 = min(c0 + CH, hi - lo)
                qc = q[:, c0:c1]
                qi = qi_all[c0:c1].view(-1, 1)
                mask = ki <= qi
                if lt == "sliding_attention" and W is not None:
                    mask = mask & ((qi - ki) < W)
                sc = torch.matmul(qc, krep.transpose(1, 2)) * scale
                sc = sc.masked_fill(~mask.view(1, c1 - c0, S), float("-inf"))
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[c0:c1] = torch.matmul(pr, vrep).transpose(0, 1)
                del sc, pr
            o = o_all.reshape(-1, nH * D)
            outs[lo:hi] = F.linear(o, attn.o_proj.weight)
            del o, krep, vrep, o_all"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
import ast
ast.parse(s)
print("attention chunking added")
