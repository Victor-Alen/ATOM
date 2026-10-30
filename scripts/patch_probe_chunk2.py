#!/usr/bin/env python3
"""Fix the probe's chunk block for its 4-D (batch-first) tensors."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
start = s.index("            rep = nH // nKV")
end = s.index("            attn_out = F.linear(o, attn.o_proj.weight)")
new = '''            rep = nH // nKV
            kk = k.repeat_interleave(rep, dim=1)   # [1, nH, T, D]
            vv = v.repeat_interleave(rep, dim=1)
            ki = torch.arange(T, device=dev).view(1, -1)
            CH = 512
            o_all = torch.empty(1, nH, T, D, dtype=q.dtype, device=dev)
            for c0 in range(0, T, CH):
                c1 = min(c0 + CH, T)
                qi = pos[c0:c1].view(-1, 1)
                maskc = ki <= qi
                if attn.sliding_window is not None and args.window_rule != "none":
                    if args.window_rule == "lt":
                        maskc = maskc & ((qi - ki) < W)
                    else:
                        maskc = maskc & ((qi - ki) <= W)
                sc = torch.matmul(q[:, :, c0:c1], kk.transpose(2, 3)) * scaling
                sc = sc.masked_fill(~maskc.view(1, 1, c1 - c0, T), float("-inf"))
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, :, c0:c1] = torch.matmul(pr, vv)
                del sc, pr
            o = o_all.transpose(1, 2).contiguous().view(1, T, nH * D)
'''
s = s[:start] + new + s[end:]
p.write_text(s)
ast.parse(s)
print("probe chunk block fixed for 4-D tensors")
