#!/usr/bin/env python3
"""Add a fourth variant to the triple probe: ATOM's own triton_kernel_moe_forward
(routing computed inside triton), to separate the routing bridge from the kernel."""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

# 1) pass router_logits into the probe
old_call = (
    "            return _atom_moe_triple(" + NL
    + "                layer, x, topk_weights, topk_ids, activation, expert_map," + NL
    + "                __import__('os').environ['ATOM_MOE_TRIPLE']," + NL
    + "            )" + NL
)
new_call = (
    "            return _atom_moe_triple(" + NL
    + "                layer, x, topk_weights, topk_ids, activation, expert_map," + NL
    + "                __import__('os').environ['ATOM_MOE_TRIPLE'], router_logits," + NL
    + "            )" + NL
)
assert old_call in src
src = src.replace(old_call, new_call, 1)

# 2) signature
old_sig = "def _atom_moe_triple(layer, x, topk_weights, topk_ids, activation, expert_map, path):"
new_sig = (
    "def _atom_moe_triple(layer, x, topk_weights, topk_ids, activation, expert_map, path,"
    + " router_logits=None):"
)
assert old_sig in src
src = src.replace(old_sig, new_sig, 1)

# 3) add the fourth variant right after the triton list
anchor = "    _ref = [" + NL
variant = (
    "    _direct = []" + NL
    + "    try:" + NL
    + "        from atom.model_ops.fused_moe_triton import triton_kernel_moe_forward as _tkmf" + NL
    + "        for _ in range(2):" + NL
    + "            _direct.append(_tkmf(" + NL
    + "                x, layer._triton_w13, layer._triton_w2, router_logits," + NL
    + "                topk=int(topk_weights.shape[1]), renormalize=True," + NL
    + "                activation='silu', global_num_experts=layer.w13_weight.shape[0]," + NL
    + "                expert_map=expert_map," + NL
    + "            ))" + NL
    + "    except Exception as _exc:" + NL
    + "        _direct = [x]" + NL
    + "        _rec_dbg['direct_err'] = f'{type(_exc).__name__}: {_exc}'" + NL
    + "    _ref = [" + NL
)
assert anchor in src
src = src.replace(anchor, variant, 1)

# 4) record the delta
old_rec_tail = "        'triton_spread': max(_mad(_trit[0], t) for t in _trit)," + NL
new_rec_tail = (
    "        'triton_spread': max(_mad(_trit[0], t) for t in _trit)," + NL
    + "        'direct_distinct': len({_dg(t) for t in _direct})," + NL
    + "        'd_direct_vs_ref': _mad(_direct[0], _ref[0])," + NL
)
assert old_rec_tail in src
src = src.replace(old_rec_tail, new_rec_tail, 1)

F.write_text(src)
ast.parse(src)
print("fourth variant added")
