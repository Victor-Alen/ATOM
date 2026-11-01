#!/usr/bin/env python3
"""Record the inputs alongside the layers, so run-to-run differences can be attributed
to nondeterminism rather than to different requests."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    _LDUMP = {}
    _dump_dir = os.environ.get("ATOM_REFERENCE_DUMPL")"""
new = """    _LDUMP = {
        "ids_hash": int(torch.tensor(input_ids.tolist(), dtype=torch.int64).sum().item()),
        "n_new": int(input_ids.numel()),
        "first_pos": int(positions[0].item()) if positions.numel() else -1,
        "moe_det": os.environ.get("ATOM_REFERENCE_MOE_DET", "0"),
        "emb_last": x[-1].detach().float().cpu() if int(x.shape[0]) > 1 else None,
    }
    _dump_dir = os.environ.get("ATOM_REFERENCE_DUMPL")"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("dump inputs recorded")
