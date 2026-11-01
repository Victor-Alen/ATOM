#!/usr/bin/env python3
"""Record the batch structure (cu_seqlens_q, context_lens, counts) so the extra token
can be attributed to a specific request slot."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        "pos_head": positions[:4].detach().cpu().tolist() if positions.numel() else [],"""
new = """        "pos_head": positions[:4].detach().cpu().tolist() if positions.numel() else [],
        "pos_tail": positions[-4:].detach().cpu().tolist() if positions.numel() else [],
        "cu_q": cu_q.tolist() if hasattr(cu_q, "tolist") else list(cu_q),
        "ctx_lens_raw": (
            md.context_lens.tolist() if hasattr(md.context_lens, "tolist") else None
        ),
        "n_positions": int(positions.numel()),"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("batch structure recorded")
