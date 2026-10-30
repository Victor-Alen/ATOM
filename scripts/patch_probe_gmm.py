#!/usr/bin/env python3
"""Probe: use HF's own grouped_mm MoE path (the one HF actually dispatches to).

This isolates "is the rest of my forward bit-exact?" from the MoE, which we will
re-implement inside ATOM separately.
"""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
if "grouped_mm_experts_forward" not in s:
    s = s.replace(
        "from transformers import AutoTokenizer",
        "from transformers import AutoTokenizer" + chr(10) + "from transformers.integrations.moe import grouped_mm_experts_forward",
        1,
    )
    start = s.index("            final = torch.zeros_like(routed)")
    end = s.index("            moe_out = final.view(1, T, cfg.hidden_size)")
    new_block = (
        "            moe_out = grouped_mm_experts_forward(\n"
        "                layer.mlp.experts, routed, topi, topv\n"
        "            ).view(1, T, cfg.hidden_size)\n"
    )
    s = s[:start] + new_block + s[end + len("            moe_out = final.view(1, T, cfg.hidden_size)" + chr(10)):]
    p.write_text(s)
    import ast
    ast.parse(s)
    print("probe now uses HF's grouped_mm MoE")
else:
    print("already patched")
