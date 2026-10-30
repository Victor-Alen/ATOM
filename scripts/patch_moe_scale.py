#!/usr/bin/env python3
"""Probe v7: magnitude-scaling test on the layer-2 prefill input.

For scales 1, 2^-3 and 2^-6, compare the Triton path against the fp32 reference
and report absolute and relative error, plus per-row error at scale 1.
"""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

func = '''def _atom_moe_scale_test(layer, x, topk_weights, topk_ids, activation, expert_map, path):
    import hashlib as _hl
    import json as _json
    import torch as _torch

    def _mad(a, b):
        return float((a.float() - b.float()).abs().max())

    _rec = {'layer': str(getattr(layer, 'layer_name', '?')), 'rows': int(x.shape[0]),
            'scales': []}
    for _scale in (1.0, 0.125, 0.015625):
        _xs = (x.float() * _scale).to(x.dtype)
        _ref = _atom_moe_torch_ref(layer, _xs, topk_weights, topk_ids)
        _tri = _atom_triton_moe_apply(layer, _xs, topk_weights, topk_ids, activation, -1, expert_map)
        _ait = fused_moe(hidden_states=_xs, w1=layer.w13_weight, w2=layer.w2_weight,
                         topk_weight=topk_weights, topk_ids=topk_ids,
                         expert_mask=expert_map, activation=activation)
        _refmax = float(_ref.float().abs().max())
        _d_tri = _mad(_tri, _ref)
        _d_ait = _mad(_ait, _ref)
        _per_row = [float((_tri.float()[i] - _ref.float()[i]).abs().max()) for i in range(_ref.shape[0])]
        _rec['scales'].append({
            'scale': _scale,
            'ref_absmax': _refmax,
            'triton_absmax': float(_tri.float().abs().max()),
            'aiter_absmax': float(_ait.float().abs().max()),
            'd_triton_vs_ref': _d_tri,
            'rel_triton': _d_tri / _refmax if _refmax else 0.0,
            'd_aiter_vs_ref': _d_ait,
            'rel_aiter': _d_ait / _refmax if _refmax else 0.0,
            'per_row_d_triton': _per_row,
        })
    _torch.cuda.synchronize()
    with open(path, 'a') as _fh:
        _fh.write(_json.dumps(_rec) + chr(10))
    return _atom_moe_torch_ref(layer, x, topk_weights, topk_ids)


'''

if '_atom_moe_scale_test' in src:
    print('already patched')
    raise SystemExit

# insert the function right before the triple probe helper
idx = src.index('def _atom_moe_triple(')
src = src[:idx] + func + src[idx:]

hook_anchor = "        if (__import__('os').environ.get('ATOM_MOE_TRIPLE')" + NL
hook = (
    "        if (__import__('os').environ.get('ATOM_MOE_SCALE')" + NL
    + "                and '.layers.2.' in str(getattr(layer, 'layer_name', ''))" + NL
    + "                and x.shape[0] == 10):" + NL
    + "            return _atom_moe_scale_test(" + NL
    + "                layer, x, topk_weights, topk_ids, activation, expert_map," + NL
    + "                __import__('os').environ['ATOM_MOE_SCALE']," + NL
    + "            )" + NL
    + NL
)
assert hook_anchor in src
src = src.replace(hook_anchor, hook + hook_anchor, 1)

F.write_text(src)
ast.parse(src)
print("probe v7 installed")
