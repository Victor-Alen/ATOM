#!/usr/bin/env python3
"""Fix the leftover duplicate del and the double selection."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
s = s.replace("""    for h in handles:
        h.remove()
    del out
    torch.cuda.empty_cache()""", """    for h in handles:
        h.remove()
    torch.cuda.empty_cache()""", 1)
s = s.replace('print(f"argmax mismatches: {int((mine[sel].argmax(-1) != hf_logits[sel].argmax(-1)).sum())}/{len(sel)}")',
              'print(f"argmax mismatches: {int((mine.argmax(-1) != hf_logits.argmax(-1)).sum())}/{mine.shape[0]}")', 1)
p.write_text(s)
ast.parse(s)
print("probe selection fixed")
