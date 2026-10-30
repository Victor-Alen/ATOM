#!/usr/bin/env python3
"""Revert the blockwise softmax: it crashed on gfx942; the plain form is proven at
4096/6144 and the 7881 run happens on the A100 (80 GB) instead."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
start = s.index("                sc.masked_fill_(~maskc.view(1, 1, c1 - c0, T), float(\"-inf\"))")
end = s.index("                del sc, pr")
new = """                sc = sc.masked_fill(~maskc.view(1, 1, c1 - c0, T), float("-inf"))
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, :, c0:c1] = torch.matmul(pr, vv)
"""
s = s[:start] + new + s[end:]
p.write_text(s)
ast.parse(s)
print("probe reverted to the plain softmax")
