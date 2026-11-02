#!/usr/bin/env python3
"""Add a 4-D attention path that mirrors HF's batch structure (nH query heads)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

anchor = """                qc = qh[:, :, c0:c1, :].reshape(nKV, rep * (c1 - c0), D)"""
assert anchor in s, "anchor missing"
new = '''                if os.environ.get("ATOM_REFERENCE_ATTN4D") == "1":
                    q4 = qh[:, :, c0:c1, :]
                    k4e = kk.repeat_interleave(rep, dim=0).unsqueeze(0)
                    v4e = vv.repeat_interleave(rep, dim=0).unsqueeze(0)
                    sc = torch.matmul(q4, k4e.transpose(2, 3)) * scale
                    neg4 = (~mask).unsqueeze(0).unsqueeze(0).expand(1, nH, c1 - c0, ctx)
                    sc.masked_fill_(neg4, float("-inf"))
                    del neg4
                    pr32 = F.softmax(sc, dim=-1, dtype=torch.float32)
                    del sc
                    pr = pr32.to(q.dtype)
                    del pr32
                    o4 = torch.matmul(pr, v4e)
                    del pr
                    o_all[:, :, c0:c1, :] = o4[0].reshape(nKV, rep, c1 - c0, D)
                    del o4
                    continue
                qc = qh[:, :, c0:c1, :].reshape(nKV, rep * (c1 - c0), D)'''
s = s.replace(anchor, new, 1)
p.write_text(s)
ast.parse(s)
print("4-D attention path added")
