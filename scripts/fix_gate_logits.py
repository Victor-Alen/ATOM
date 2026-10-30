#!/usr/bin/env python3
"""HF Mellum returns only the last position by default; ask for all rows."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/teacher_forced_gate.py")
s = p.read_text()
old = """    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits[0] if hasattr(out, "logits") else out[0]"""
new = """    with torch.no_grad():
        try:
            out = model(input_ids=ids, logits_to_keep=int(ids.shape[1]))
        except TypeError:
            out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logits = logits[0] if logits.dim() == 3 else logits"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("logits_to_keep fixed")
