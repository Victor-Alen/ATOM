#!/usr/bin/env python3
"""Dump layer-0 q (post-norm, post-rope) and the per-head attention output."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old1 = """            q = _hf_norm(q, attn.q_norm.weight, eps)"""
new1 = """            _a0 = _dump_call and li == 0
            q = _hf_norm(q, attn.q_norm.weight, eps)
            if _a0:
                _LDUMP["q0_norm"] = q.detach().to(torch.bfloat16).cpu()"""
assert old1 in s
s = s.replace(old1, new1, 1)

old2 = """            k = k * cos + _rotate_half(k) * sin"""
new2 = """            k = k * cos + _rotate_half(k) * sin
            if _a0:
                _LDUMP["q0_rope"] = q.detach().to(torch.bfloat16).cpu()
                _LDUMP["k0_rope"] = k.detach().to(torch.bfloat16).cpu()"""
assert old2 in s
s = s.replace(old2, new2, 1)

old3 = """            o = o_all.permute(2, 0, 1, 3).reshape(n_new, nH * D)"""
new3 = """            o = o_all.permute(2, 0, 1, 3).reshape(n_new, nH * D)
            if _a0:
                _LDUMP["o0_pre_o"] = o.detach().to(torch.bfloat16).cpu()"""
assert old3 in s
s = s.replace(old3, new3, 1)
p.write_text(s)
ast.parse(s)
print("attention dumps added")
