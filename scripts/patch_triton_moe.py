#!/usr/bin/env python3
"""Env-gated Triton MoE expert path (v2: correct [E,K,N] / [E,I,H] weight layout)."""
import ast, subprocess
from pathlib import Path

WT = Path("/workspace/ATOM-tritonmoe")
subprocess.run(["git", "checkout", "--", "atom/model_ops/moe.py", "atom/model_ops/sampler.py"],
               cwd=WT, check=True)
F = WT / "atom/model_ops/moe.py"
src = F.read_text()
NL = chr(10)

anchor = "        if self.fused_experts:" + NL
idx = src.index(anchor)
branch = (
    "        if __import__('os').environ.get('ATOM_TRITON_MOE_EXPERTS') == '1':" + NL
    + "            return _atom_triton_moe_apply(" + NL
    + "                layer, x, topk_weights, topk_ids, activation," + NL
    + "                global_num_experts, expert_map," + NL
    + "            )" + NL
    + NL
)
src = src[:idx] + branch + src[idx:]

helper = NL + NL + '''def _atom_triton_moe_apply(
    layer, x, topk_weights, topk_ids, activation, global_num_experts, expert_map
):
    """Expert path through ATOM's Triton matmul_ogs kernels.

    aiter ships MoE weights in a (16,16) shuffled layout and stores them as
    [E, out, in]; the Triton kernels want [E, K, N] (w1) and [E, I, H] (w2), so
    the shuffle is inverted and the last two dims transposed once per layer.
    """
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

    _n_tot = (
        global_num_experts
        if isinstance(global_num_experts, int) and global_num_experts > 0
        else layer.w13_weight.shape[0]
    )
    _act = activation
    if not isinstance(_act, str):
        _act = getattr(_act, 'name', str(_act)).lower()
        if '.' in _act:
            _act = _act.split('.')[-1]

    _routing, _gather, _scatter = routing_from_topk(topk_weights, topk_ids, _n_tot)
    _out = _torch.empty_like(x)
    return triton_kernel_fused_experts(
        _out,
        x,
        layer._triton_w13,
        layer._triton_w2,
        _routing,
        _gather,
        _scatter,
        topk=int(topk_weights.shape[1]),
        activation=_act,
        w1_bias=None,
        w2_bias=None,
        apply_router_weight_on_input=False,
        global_num_experts=_n_tot,
        expert_map=expert_map,
    )
'''
src = src + helper
F.write_text(src)
ast.parse(src)

S = WT / "atom/model_ops/sampler.py"
ssrc = S.read_text()
sanchor = "        # No Top-K Top-P parameters, perform temperature-based sampling"
sblock = (
    "        # Temperature 0 must be a true argmax: the plain-temperature path" + NL
    + "        # below runs a Gumbel race with fresh noise on every call." + NL
    + "        if all_greedy:" + NL
    + "            return logits.argmax(dim=-1).to(torch.int)" + NL
    + NL
)
ssrc = ssrc.replace(sanchor, sblock + sanchor, 1)
S.write_text(ssrc)
ast.parse(ssrc)
print("patched v2")
