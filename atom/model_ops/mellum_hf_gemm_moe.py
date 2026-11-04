"""Native grouped/batched GEMM expert backend for production Mellum inputs."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.integrations.moe import (
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
)

from atom.utils.forward_context import get_forward_context


def unshuffle(weight):
    e, n, k = weight.shape
    return (
        weight.view(e, n // 16, k // 32, 4, 16, 8)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .view(e, n, k)
    )


def apply(layer, x, router_logits):
    assert layer.tp_size == layer.dp_size == layer.ep_size == 1
    assert x.dtype == router_logits.dtype == torch.bfloat16
    cached = getattr(layer, "_mellum_hf_experts", None)
    if cached is None:
        from atom.model_ops.mellum_piecewise import gate

        cached = SimpleNamespace(
            gate_up_proj=unshuffle(layer.w13_weight),
            down_proj=unshuffle(layer.w2_weight),
            num_experts=layer.w13_weight.shape[0],
            has_bias=False,
            is_transposed=False,
            _apply_gate=gate,
        )
        layer._mellum_hf_experts = cached
    probabilities = F.softmax(router_logits, dtype=torch.float32, dim=-1)
    weights, ids = probabilities.topk(layer.top_k, dim=-1)
    if layer.renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    weights = weights.to(router_logits.dtype)
    is_prefill = get_forward_context().context.is_prefill
    # Native HF generate changes experts implementation after prefill.
    function = grouped_mm_experts_forward if is_prefill else batched_mm_experts_forward
    # Decode contains independent matrix-vector products. Batch their
    # submission while retaining each expert GEMM's N=1 reduction extent.
    if (
        not is_prefill
        and __import__("os").environ.get("ATOM_MELLUM_BATCH_DECODE_MOE") == "1"
    ):
        # Bound selected-weight materialization at 64 token rows. Larger
        # scheduler batches keep the same independent GEMVs in more chunks.
        if x.shape[0] <= 64:
            return batched_mm_experts_forward(cached, x, ids, weights)
        return torch.cat(
            [
                batched_mm_experts_forward(
                    cached, x[a : a + 64], ids[a : a + 64], weights[a : a + 64]
                )
                for a in range(0, x.shape[0], 64)
            ],
            dim=0,
        )
    if __import__("os").environ.get("ATOM_MELLUM_REQUEST_GEMM") == "1":
        from atom.model_ops.mellum_request_gemm import ranges

        parts = ranges(x)
        return torch.cat(
            [function(cached, x[a:b], ids[a:b], weights[a:b]) for a, b in parts], dim=0
        )
    return function(cached, x, ids, weights)
