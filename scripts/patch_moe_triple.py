#!/usr/bin/env python3
"""Add an env-gated three-way MoE comparison to the isolated worktree.

ATOM_MOE_TRIPLE=<path> computes, for the first planned layer only:
  (a) aiter fused_moe (2-stage CK, atomic accumulate)  x3
  (b) Triton matmul_ogs path (ATOM_TRITON_MOE_EXPERTS implementation)  x3
  (c) per-expert PyTorch fp32 reference with inverted shuffle  x2
and logs pairwise max-abs differences plus digests.
"""
import ast, subprocess
from pathlib import Path

WT = Path("/workspace/ATOM-tritonmoe")
subprocess.run(["git", "checkout", "--", "atom/model_ops/moe.py"], cwd=WT, check=True)
src = (WT / "atom/model_ops/sampler.py").read_text()
NL = chr(10)
F = WT / "atom/model_ops/moe.py"
body = F.read_text()

# re-apply the triton branch (wip commit content is gone after checkout)
anchor = "        if self.fused_experts:" + NL
idx = body.index(anchor)
branch = (
    "        if __import__('os').environ.get('ATOM_TRITON_MOE_EXPERTS') == '1':" + NL
    + "            return _atom_triton_moe_apply(" + NL
    + "                layer, x, topk_weights, topk_ids, activation," + NL
    + "                global_num_experts, expert_map," + NL
    + "            )" + NL
    + NL
)
body = body[:idx] + branch + body[idx:]

helper = NL + NL + '''def _atom_triton_moe_apply(
    layer, x, topk_weights, topk_ids, activation, global_num_experts, expert_map
):
    """Expert path through ATOM's Triton matmul_ogs kernels."""
    import torch as _torch

    from atom.model_ops.fused_moe_triton import (
        routing_from_topk,
        triton_kernel_fused_experts,
    )

    if not getattr(layer, '_triton_moe_ready', False):
        with _torch.no_grad():
            def _unshuffle(w):
                E, n, k = w.shape
                assert n % 16 == 0 and k % 32 == 0, w.shape
                return (
                    w.view(E, n // 16, k // 32, 4, 16, 8)
                    .permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .view(E, n, k)
                )

            layer._triton_w13 = _unshuffle(layer.w13_weight).transpose(-1, -2).contiguous()
            layer._triton_w2 = _unshuffle(layer.w2_weight).transpose(-1, -2).contiguous()
            layer._triton_moe_ready = True

    _n_tot = layer.w13_weight.shape[0]
    _act = activation
    if not isinstance(_act, str):
        _act = getattr(_act, 'name', str(_act)).lower()
        if '.' in _act:
            _act = _act.split('.')[-1]
    _routing, _gather, _scatter = routing_from_topk(topk_weights, topk_ids, _n_tot)
    _out = _torch.empty_like(x)
    return triton_kernel_fused_experts(
        _out, x, layer._triton_w13, layer._triton_w2, _routing, _gather, _scatter,
        topk=int(topk_weights.shape[1]), activation=_act, w1_bias=None, w2_bias=None,
        apply_router_weight_on_input=False, global_num_experts=_n_tot, expert_map=expert_map,
    )


def _atom_moe_torch_ref(layer, x, topk_weights, topk_ids):
    """Per-expert fp32 reference on the unshuffled weights, fixed accumulation order."""
    import torch as _torch
    import torch.nn.functional as _F

    _w13 = layer._triton_w13.transpose(-1, -2).float()  # [E, 2I, H]
    _w2 = layer._triton_w2.transpose(-1, -2).float()    # [E, H, I]
    _n, _slots = topk_ids.shape
    _acc = _torch.zeros(x.shape[0], x.shape[1], dtype=_torch.float32, device=x.device)
    _xf = x.float()
    for _t in range(_n):
        _row = _torch.zeros(x.shape[1], dtype=_torch.float32, device=x.device)
        for _s in range(_slots):
            _e = int(topk_ids[_t, _s])
            _g = _F.linear(_xf[_t:_t + 1], _w13[_e])
            _gate, _up = _g.chunk(2, dim=-1)
            _inter = _F.silu(_gate) * _up
            _o = _F.linear(_inter, _w2[_e])
            _row = _row + _o[0] * float(topk_weights[_t, _s])
        _acc[_t] = _row
    return _acc.to(x.dtype)


def _atom_moe_triple(layer, x, topk_weights, topk_ids, activation, expert_map, path):
    """Compare aiter, Triton matmul_ogs and the fp32 per-expert reference."""
    import hashlib as _hl
    import json as _json
    import os as _os
    import torch as _torch

    def _dg(t):
        return _hl.sha1(
            t.detach().contiguous().view(_torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()[:12]

    def _mad(a, b):
        return float((a.float() - b.float()).abs().max())

    _aiter = [
        fused_moe(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
                  topk_weight=topk_weights, topk_ids=topk_ids,
                  expert_mask=expert_map, activation=activation)
        for _ in range(3)
    ]
    _trit = [
        _atom_triton_moe_apply(layer, x, topk_weights, topk_ids, activation, -1, expert_map)
        for _ in range(3)
    ]
    _ref = [
        _atom_moe_torch_ref(layer, x, topk_weights, topk_ids) for _ in range(2)
    ]
    _rec = {
        'layer': getattr(layer, 'layer_name', '?'),
        'rows': int(x.shape[0]),
        'topk_ids_head': [int(v) for v in topk_ids[0].tolist()],
        'aiter_digests': [_dg(t) for t in _aiter],
        'triton_digests': [_dg(t) for t in _trit],
        'ref_digests': [_dg(t) for t in _ref],
        'aiter_distinct': len({_dg(t) for t in _aiter}),
        'triton_distinct': len({_dg(t) for t in _trit}),
        'ref_distinct': len({_dg(t) for t in _ref}),
        'maxabs_ref': float(_ref[0].float().abs().max()),
        'd_triton_vs_ref': _mad(_trit[0], _ref[0]),
        'd_aiter_vs_ref': _mad(_aiter[0], _ref[0]),
        'd_triton_vs_aiter': _mad(_trit[0], _aiter[0]),
        'aiter_spread': max(_mad(_aiter[0], t) for t in _aiter),
        'triton_spread': max(_mad(_trit[0], t) for t in _trit),
    }
    _torch.cuda.synchronize()
    with open(path, 'a') as _fh:
        _fh.write(_json.dumps(_rec) + chr(10))
    return _trit[0]
'''
body = body + helper

# hook: intercept layer 0's MoE when ATOM_MOE_TRIPLE is set
hook_anchor = "        if __import__('os').environ.get('ATOM_TRITON_MOE_EXPERTS') == '1':" + NL
hook = (
    "        if (__import__('os').environ.get('ATOM_MOE_TRIPLE')" + NL
    + "                and '.layers.0.' in str(getattr(layer, 'layer_name', ''))" + NL
    + "                and x.shape[0] <= 64):" + NL
    + "            return _atom_moe_triple(" + NL
    + "                layer, x, topk_weights, topk_ids, activation, expert_map," + NL
    + "                __import__('os').environ['ATOM_MOE_TRIPLE']," + NL
    + "            )" + NL
    + NL
)
idx2 = body.index(hook_anchor)
body = body[:idx2] + hook + body[idx2:]

F.write_text(body)
ast.parse(body)
print("triple probe installed")
