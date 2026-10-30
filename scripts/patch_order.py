#!/usr/bin/env python3
"""Order bug: ctx_slots were derived from the single-token positions before the
recompute path replaced positions/cu_q with the full sequence, so attention ran
over one key. Move the recompute block above the ctx_slots derivation."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

start = s.index('    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":')
end = s.index("    x = F.embedding(input_ids, model.model.embed_tokens.weight)")
block = s[start:end]
s = s[:start] + s[end:]

anchor = "    ctx_slots = []"
assert anchor in s
s = s.replace(anchor, block + anchor, 1)
p.write_text(s)
ast.parse(s)
print("recompute block moved before ctx_slots")
