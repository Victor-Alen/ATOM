#!/usr/bin/env python3
"""Extend the latched L0 dump with the full K/V, the last query row, the attention
output before o_proj, and the window, so the attention step can be replayed offline."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """            if _do_dump:
                torch.save(
                    {
                        "q_last": q[-1].detach().float().cpu(),
                        "k_last": k[-1].detach().float().cpu(),
                        "cos_last": cos[-1].detach().float().cpu(),
                        "sin_last": sin[-1].detach().float().cpu(),
                    },
                    f"{_d1}/l0_qk.pt",
                )"""
new = """            if _do_dump:
                torch.save(
                    {
                        "q_last": q[-1].detach().float().cpu(),
                        "k_last": k[-1].detach().float().cpu(),
                        "cos_last": cos[-1].detach().float().cpu(),
                        "sin_last": sin[-1].detach().float().cpu(),
                        "k_all": k.detach().to(torch.bfloat16).cpu(),
                        "v_all": v.detach().to(torch.bfloat16).cpu(),
                        "n_heads": nH,
                        "n_kv": nKV,
                        "head_dim": D,
                        "window": -1 if W is None else int(W),
                        "scale": float(scale),
                    },
                    f"{_d1}/l0_qk.pt",
                )"""
assert old in s
s = s.replace(old, new, 1)

old2 = """            o = o_all.permute(2, 0, 1, 3).reshape(n_new, nH * D)"""
new2 = """            o = o_all.permute(2, 0, 1, 3).reshape(n_new, nH * D)
            if _do_dump:
                torch.save(o[-1].detach().float().cpu(), f"{_d1}/l0_o_last.pt")"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("attention replay dump extended")
