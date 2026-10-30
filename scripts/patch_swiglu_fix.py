#!/usr/bin/env python3
"""The Triton MoE was clamping the activation at swiglu_limit=7.0 (ATOM default).
Mellum has no such limit, so pass a huge limit to disable the clamp."""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

count = 0
old_call = (
    "        apply_router_weight_on_input=False, global_num_experts=_n_tot, expert_map=expert_map," + NL
    + "    )"
)
new_call = (
    "        apply_router_weight_on_input=False, global_num_experts=_n_tot, expert_map=expert_map," + NL
    + "        swiglu_limit=1e30," + NL
    + "    )"
)
if old_call in src:
    src = src.replace(old_call, new_call, 1)
    count += 1

# also the direct variant used by the probe
old_direct = (
    "                activation='silu', global_num_experts=layer.w13_weight.shape[0]," + NL
    + "                expert_map=expert_map," + NL
)
new_direct = (
    "                activation='silu', global_num_experts=layer.w13_weight.shape[0]," + NL
    + "                expert_map=expert_map, swiglu_limit=1e30," + NL
)
if old_direct in src:
    src = src.replace(old_direct, new_direct, 1)
    count += 1

F.write_text(src)
ast.parse(src)
print(f"patched {count} call sites with swiglu_limit=1e30")
