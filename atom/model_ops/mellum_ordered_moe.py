"""Deterministic CK MoE combine for Mellum's unquantized bf16 path.

The AITER stage1 and stage2 GEMM selections are unchanged. Stage2 writes one
bf16 row per routed slot, avoiding expert collisions in its AtomicAdd output.
The eight rows are then accumulated in a fixed fp32 order.
"""

import torch
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting


def mellum_ordered_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: ActivationType,
) -> torch.Tensor:
    if (
        hidden_states.dtype != torch.bfloat16
        or w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
    ):
        raise ValueError("Mellum ordered MoE requires unquantized bf16 tensors")
    if activation != ActivationType.Silu:
        raise ValueError("Mellum ordered MoE requires SiLU activation")

    tokens, topk = topk_ids.shape
    if tokens == 0:
        return hidden_states
    experts, _, _ = w13.shape
    model_dim, inter_dim = w2.shape[1:]
    if topk != 8 or w2.shape[0] != experts or hidden_states.shape != (tokens, model_dim):
        raise ValueError("Unexpected Mellum MoE shape")

    metadata = get_2stage_cfgs(
        get_padded_M(tokens), model_dim, inter_dim, experts, topk,
        hidden_states.dtype, w13.dtype, w13.dtype, QuantType.No,
        inter_dim != w13.shape[1], activation, False, 0, 0,
        getattr(w13, "is_shuffled", False) or getattr(w2, "is_shuffled", False),
    )
    if metadata.run_1stage or metadata.ksplit != 0:
        raise RuntimeError("Mellum ordered MoE requires the CK two-stage, unsplit dispatch")

    sorted_ids, sorted_weights, sorted_experts, num_valid, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, hidden_states.dtype,
        metadata.block_m,
    )
    a2 = torch.empty(
        (tokens, topk, inter_dim), dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    metadata.stage1(
        hidden_states, w13, w2, sorted_ids, sorted_experts, num_valid,
        a2, topk, block_m=metadata.block_m, a1_scale=None, w1_scale=None,
        sorted_weights=None,
    )

    # AITER packs token in the low 24 bits and slot in the high 8 bits.
    # Map each valid (token, slot) to a unique output row. Padding maps to
    # an out-of-range row and is skipped by the original CK bounds checks.
    token = sorted_ids.bitwise_and(0xFFFFFF)
    slot = sorted_ids.bitwise_right_shift(24).bitwise_and(0xFF)
    valid = (token < tokens) & (slot < topk)
    flat_ids = torch.where(valid, token * topk + slot, tokens * topk)

    per_slot = torch.zeros(
        (tokens * topk, model_dim), dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    metadata.stage2(
        a2.view(tokens * topk, 1, inter_dim), w13, w2, flat_ids,
        sorted_experts, num_valid, per_slot, 1,
        block_m=metadata.block_m, w2_scale=None, a2_scale=None,
        sorted_weights=sorted_weights,
    )

    per_slot = per_slot.view(tokens, topk, model_dim)
    combined = torch.zeros(
        (tokens, model_dim), dtype=torch.float32, device=hidden_states.device,
    )
    for slot_index in range(topk):
        combined.add_(per_slot[:, slot_index, :].float())
    return combined.to(hidden_states.dtype)
