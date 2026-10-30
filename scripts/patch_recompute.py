#!/usr/bin/env python3
"""Add ATOM_REFERENCE_RECOMPUTE=1: rebuild the whole sequence each step.

Keeps the token history per request (keyed by its first physical block, which is
stable and unique among live requests) and runs the full forward from scratch, so
the incremental KV path cannot be the source of a wrong token.
"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

if "_TOKENS" not in s:
    s = s.replace("_KV = {}", "_KV = {}\n_TOKENS = {}", 1)
    old = """    x = F.embedding(input_ids, model.model.embed_tokens.weight)"""
    new = """    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        key = int(bs_bt[0, 0].item())
        hist = _TOKENS.setdefault(key, [])
        new_ids = input_ids.tolist()
        if len(hist) >= len(new_ids):
            hist[:] = new_ids
        else:
            hist.extend(new_ids[len(hist):])
        input_ids = torch.tensor(hist, dtype=torch.long, device=dev)
        positions = torch.arange(len(hist), device=dev)
        keep = torch.arange(len(hist) - len(new_ids), len(hist), device=dev)
        cu_q = [0, len(hist)]
        bs = 1
        ctx_lens = [len(hist)]

    x = F.embedding(input_ids, model.model.embed_tokens.weight)"""
    assert old in s
    s = s.replace(old, new, 1)
    s = s.replace("    return _hf_norm(x, model.model.norm.weight, eps)",
                  """    out = _hf_norm(x, model.model.norm.weight, eps)
    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        return out[keep]
    return out""", 1)
    # ctx_slots / write slots must follow the recomputed positions
    p.write_text(s)
    ast.parse(s)
    print("recompute path added")
else:
    print("already present")
