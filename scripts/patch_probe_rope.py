#!/usr/bin/env python3
"""Use HF's model-level per-layer-type rotary (YaRN for full-attention layers)."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
old = """            cos, sin = attn.rotary_emb(v, position_ids=pos[None, :])  # HF's own rope
            q = q * cos[:, :, None, :] + rotate_half(q) * sin[:, :, None, :]
            k = k * cos[:, :, None, :] + rotate_half(k) * sin[:, :, None, :]"""
new = """            lt = cfg.layer_types[li]
            cos, sin = model.model.rotary_emb(x, pos[None, :], lt)  # HF's own, per layer type
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
import ast
ast.parse(s)
print("probe now uses HF's per-layer-type rotary embedding")
