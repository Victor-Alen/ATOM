#!/usr/bin/env python3
"""Dump the last row after attention and after the MLP for the prefill call, so the
first differing layer can be found against HF's hidden_states."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """        x = x + outs

        h2 = _hf_norm(x, layer.post_attention_layernorm.weight, eps)"""
new = """        x = x + outs
        _ldir = os.environ.get("ATOM_REFERENCE_DUMPL")
        if _ldir and n_new > 1:
            torch.save(x[-1].detach().float().cpu(), f"{_ldir}/attn_L{li}.pt")

        h2 = _hf_norm(x, layer.post_attention_layernorm.weight, eps)"""
assert old in s
s = s.replace(old, new, 1)

old2 = """        x = x + moe
        if os.environ.get("ATOM_REFERENCE_DEBUG") == "2" and li < 8:"""
new2 = """        x = x + moe
        if _ldir and n_new > 1:
            torch.save(x[-1].detach().float().cpu(), f"{_ldir}/blk_L{li}.pt")
        if os.environ.get("ATOM_REFERENCE_DEBUG") == "2" and li < 8:"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("per-layer prefill dumps added")
