#!/usr/bin/env python3
"""Dump the raw embedding row of the prefill so it can be compared with HF's
hidden_states[0] exactly (the earlier x-outs reconstruction was not exact)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    x = F.embedding(input_ids, model.model.embed_tokens.weight)"""
new = """    x = F.embedding(input_ids, model.model.embed_tokens.weight)
    _ldir0 = os.environ.get("ATOM_REFERENCE_DUMPL")
    if _ldir0 and int(x.shape[0]) > 1:
        torch.save(x[-1].detach().float().cpu(), f"{_ldir0}/emb.pt")
        torch.save(
            torch.tensor(input_ids[-8:].tolist(), dtype=torch.long), f"{_ldir0}/last8.pt"
        )"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("embedding dump added")
