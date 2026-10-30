#!/usr/bin/env python3
"""Keep only the reference rows of the logits: at 7881 tokens the full logits
tensor is 3 GB in fp32 and is the difference between fitting and OOM."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
old = "    hf_logits = out.logits[0].float()"
new = ("    sel = torch.arange(P - 1, P + len(ref_tokens) - 1, device=args.device)\n"
       "    hf_logits = out.logits[0][sel].float()\n"
       "    del out\n"
       "    logits = None")
assert old in s
s = s.replace(old, new, 1)

old2 = """    sel = torch.arange(P - 1, P + len(ref_tokens) - 1, device=dev)
    d = (mine[sel].float() - hf_logits[sel]).abs()"""
new2 = """    d = (mine.float() - hf_logits).abs()"""
if old2 in s:
    s = s.replace(old2, new2, 1)
s = s.replace("        mine = F.linear(x.view(-1, cfg.hidden_size), model.lm_head.weight).view(T, -1)",
              "        mine = F.linear(x[sel].view(-1, cfg.hidden_size), model.lm_head.weight)", 1)
p.write_text(s)
ast.parse(s)
print("probe keeps only reference logit rows")
