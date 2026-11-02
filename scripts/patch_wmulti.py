#!/usr/bin/env python3
"""Dump q/k/v weight rows for layers 0-3 to check whether non-zero layers carry the
right weights (layer 0 was already verified bit-exact)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if li == 0 and os.environ.get("ATOM_REFERENCE_WDUMP"):"""
new = """        if li < 4 and os.environ.get("ATOM_REFERENCE_WDUMP"):"""
assert old in s
s = s.replace(old, new, 1)
old2 = """            torch.save(
                {
                    "q_row0": q_w[0].detach().float().cpu(),
                    "k_row0": k_w[0].detach().float().cpu(),
                    "v_row0": v_w[0].detach().float().cpu(),
                    "fused_shape": torch.tensor(attn.qkv_proj.weight.shape),
                    "fused_contig": torch.tensor(int(attn.qkv_proj.weight.is_contiguous())),
                    "q_contig": torch.tensor(int(q_w.is_contiguous())),
                },
                _wd,
            )"""
new2 = """            _prev = {}
            if os.path.exists(_wd):
                try:
                    _prev = torch.load(_wd)
                except Exception:  # noqa: BLE001
                    _prev = {}
            _prev[f"L{li}"] = {
                "q_row0": q_w[0].detach().float().cpu(),
                "k_row0": k_w[0].detach().float().cpu(),
                "v_row0": v_w[0].detach().float().cpu(),
                "o_col0": attn.o_proj.weight[:, 0].detach().float().cpu(),
                "qn": attn.q_norm.weight.detach().float().cpu(),
                "kn": attn.k_norm.weight.detach().float().cpu(),
            }
            torch.save(_prev, _wd)"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("multi-layer weight dump added")
