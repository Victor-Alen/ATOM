#!/usr/bin/env python3
"""Layout audit: print shape/stride/contiguity of the layer-0 tensors that feed GEMMs,
so they can be compared with HF's equivalents."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

helper = '''

def _lay(tag, t):
    import os as _o

    if _o.environ.get("ATOM_REFERENCE_LAYOUT") == "1":
        print(
            f"[lay] {tag}: shape={tuple(t.shape)} stride={tuple(t.stride())} "
            f"contig={t.is_contiguous()} dtype={t.dtype}",
            flush=True,
        )

'''
anchor = "def _hf_norm("
if "_lay(" not in s.split(anchor)[0]:
    s = s.replace(anchor, helper.strip("\n") + "\n\n\n" + anchor, 1)

old = """            q = F.linear(h, q_w).view(-1, nH, D)
            k = F.linear(h, k_w).view(-1, nKV, D)
            v = F.linear(h, v_w).view(-1, nKV, D)"""
new = """            _lay("h", h)
            q_lin = F.linear(h, q_w)
            k_lin = F.linear(h, k_w)
            v_lin = F.linear(h, v_w)
            _lay("q_lin", q_lin)
            _lay("k_lin", k_lin)
            q = q_lin.view(-1, nH, D)
            k = k_lin.view(-1, nKV, D)
            v = v_lin.view(-1, nKV, D)"""
if old in s:
    s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("layout audit hooks added" if "q_lin" in s else "pattern not found")
