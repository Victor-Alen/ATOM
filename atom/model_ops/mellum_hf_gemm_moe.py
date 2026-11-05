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
    if (
        __import__('os').environ.get('ATOM_MELLUM_NATIVE_HF_ROUTER') == '1'
        and layer.top_k == 8 and layer.renormalize
        and router_logits.shape[-1] == 64
    ):
        # Requires the isolated HF-order AITER router build: full FP32 softmax,
        # padded-32 tie order, adjacent-tree top-8 sum, preserved denormals.
        fused_bf16 = __import__('os').environ.get('ATOM_MELLUM_ROUTER_FUSED_BF16') == '1'
        weights = torch.empty((router_logits.shape[0], 8), device=router_logits.device,
                              dtype=router_logits.dtype if fused_bf16 else torch.float32)
        ids = torch.empty_like(weights, dtype=torch.int32)
        source_rows = torch.empty_like(ids)
        if fused_bf16:
            from aiter.jit.core import get_module
            warps = int(__import__('os').environ.get('ATOM_MELLUM_ROUTER_WARPS', '8'))
            get_module('module_moe_asm').mellum_topk_softmax_bf16(
                weights, ids, source_rows, router_logits, warps)
        else:
            from aiter import topk_softmax
            topk_softmax(weights, ids, source_rows, router_logits, True, 0, '')
            weights = weights.to(router_logits.dtype)
    else:
        probabilities = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        weights, ids = probabilities.topk(layer.top_k, dim=-1)
        if layer.renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(router_logits.dtype)
    is_prefill = get_forward_context().context.is_prefill
    if is_prefill and __import__('os').environ.get('ATOM_MELLUM_NATIVE_PREFILL_STAGE1') == '1':
        from atom.model_ops.mellum_native_prefill_moe import experts
        from atom.model_ops.mellum_request_gemm import ranges
        parts = ranges(x)
        if len(parts) == 1:
            return experts(layer, cached, x, ids, weights)
        streams = int(__import__('os').environ.get('ATOM_MELLUM_PREFILL_MOE_STREAMS', '0'))
        if streams:
            from atom.model_ops.mellum_prefill_streams import experts_parallel
            return experts_parallel(layer, cached, x, ids, weights, parts, streams)
        return torch.cat([experts(layer, cached, x[a:b], ids[a:b], weights[a:b])
                          for a, b in parts], dim=0)
    if not is_prefill and __import__('os').environ.get('ATOM_MELLUM_NATIVE_DECODE_MOE') == '1':
        from atom.model_ops.mellum_native_decode_moe import experts
        return experts(x, layer.w13_weight, layer.w2_weight, ids, weights)
    mode = __import__('os').environ.get('ATOM_MELLUM_INDEXED_DECODE_MOE', '')
    if not is_prefill and mode in ('indexed', 'grouped'):
        if mode == 'grouped':
            from atom.model_ops.mellum_grouped_decode import experts
        else:
            from atom.model_ops.mellum_indexed_bmm import experts
        if x.shape[0] <= 64:
            return experts(cached, x, ids, weights)
        return torch.cat([experts(cached, x[a:a+64], ids[a:a+64], weights[a:a+64])
                          for a in range(0, x.shape[0], 64)], dim=0)
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
