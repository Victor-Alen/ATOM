#!/usr/bin/env python3
"""Fix: build the triton weight cache before the reference; return the aiter result."""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

old = (
    "        _xs = (x.float() * _scale).to(x.dtype)" + NL
    + "        _ref = _atom_moe_torch_ref(layer, _xs, topk_weights, topk_ids)" + NL
    + "        _tri = _atom_triton_moe_apply(layer, _xs, topk_weights, topk_ids, activation, -1, expert_map)" + NL
)
new = (
    "        _xs = (x.float() * _scale).to(x.dtype)" + NL
    + "        _tri = _atom_triton_moe_apply(layer, _xs, topk_weights, topk_ids, activation, -1, expert_map)" + NL
    + "        _ref = _atom_moe_torch_ref(layer, _xs, topk_weights, topk_ids)" + NL
)
assert old in src, "loop order pattern missing"
src = src.replace(old, new, 1)

old_ret = "    return _atom_moe_torch_ref(layer, x, topk_weights, topk_ids)" + NL
new_ret = (
    "    return fused_moe(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight," + NL
    + "                     topk_weight=topk_weights, topk_ids=topk_ids," + NL
    + "                     expert_mask=expert_map, activation=activation)" + NL
)
assert old_ret in src
src = src.replace(old_ret, new_ret, 1)

F.write_text(src)
ast.parse(src)
print("scale probe fixed")
