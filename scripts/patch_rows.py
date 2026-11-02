#!/usr/bin/env python3
"""Dump the layer-0 score and softmax rows for token 93."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old1 = """                sc.masked_fill_(neg, float("-inf"))
                del neg"""
new1 = """                sc.masked_fill_(neg, float("-inf"))
                del neg
                if _a0 and c0 == 0 and (c1 - c0) > 93:
                    _LDUMP["sc93"] = sc[:, :, 93, :].detach().float().cpu()"""
assert old1 in s
s = s.replace(old1, new1, 1)

old2 = """                    pr32 = F.softmax(sc, dim=-1, dtype=torch.float32)
                    del sc
                    pr = pr32.to(q.dtype)"""
new2 = """                    pr32 = F.softmax(sc, dim=-1, dtype=torch.float32)
                    del sc
                    if _a0 and c0 == 0 and (c1 - c0) > 93:
                        _LDUMP["pr93"] = pr32[:, :, 93, :].detach().float().cpu()
                    pr = pr32.to(q.dtype)"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("score/softmax row dumps added")
