#!/usr/bin/env python3
"""reference_mode: obtain fwd_ctx internally via get_forward_context()."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
s = s.replace(
    "def reference_forward(model, input_ids, positions, fwd_ctx):",
    "def reference_forward(model, input_ids, positions):" + chr(10)
    + "    from atom.utils.forward_context import get_forward_context" + chr(10)
    + chr(10)
    + "    fwd_ctx = get_forward_context()",
    1,
)
s = s.replace("    md = fwd_ctx.attn_metadata" + chr(10) + "    cfg = model.config",
              "    md = fwd_ctx.attn_metadata" + chr(10)
              + "    if isinstance(md, dict):" + chr(10)
              + "        md = next(iter(md.values()))" + chr(10)
              + "    if __import__('os').environ.get('ATOM_REFERENCE_DEBUG') == '1':" + chr(10)
              + "        print('[ref] cu_seqlens_q', md.cu_seqlens_q.tolist()[:6], 'ctx', md.context_lens.tolist()[:6], flush=True)" + chr(10)
              + "    cfg = model.config",
              1)
p.write_text(s)
import ast
ast.parse(s)
print("reference_mode now self-contained")
