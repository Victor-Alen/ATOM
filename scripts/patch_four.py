#!/usr/bin/env python3
"""Dump the four semantics the comparison protocol asks for, layer 0, full matrices:
attn_out (post o_proj, pre-residual), post-attn residual, moe out, block out."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old1 = """            if li == 0:
                _LDUMP["attn0_full"] = outs.detach().to(torch.bfloat16).cpu()"""
new1 = """            if li == 0:
                _LDUMP["attn0_full"] = outs.detach().to(torch.bfloat16).cpu()
                _LDUMP["res_after_attn_0"] = (x + outs).detach().to(torch.bfloat16).cpu()"""
assert old1 in s
s = s.replace(old1, new1, 1)

old2 = """            _LDUMP[f"h2_{li}"] = h2[-1].detach().float().cpu()
            _LDUMP[f"moe_{li}"] = moe[-1].detach().float().cpu()"""
new2 = """            _LDUMP[f"h2_{li}"] = h2[-1].detach().float().cpu()
            _LDUMP[f"moe_{li}"] = moe[-1].detach().float().cpu()
            if li == 0:
                _LDUMP["moe0_full"] = moe.detach().to(torch.bfloat16).cpu()"""
assert old2 in s
s = s.replace(old2, new2, 1)

old3 = """        if _dump_call:
            _LDUMP[f"blk_{li}"] = x[-1].detach().float().cpu()"""
new3 = """        if _dump_call:
            _LDUMP[f"blk_{li}"] = x[-1].detach().float().cpu()
            if li == 0:
                _LDUMP["blk0_full"] = x.detach().to(torch.bfloat16).cpu()"""
assert old3 in s
s = s.replace(old3, new3, 1)
p.write_text(s)
ast.parse(s)
print("four-tensor dump added")
