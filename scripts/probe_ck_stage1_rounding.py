#!/usr/bin/env python3
"""Locate residual L0 MoE error after same-CK-stage2 ordered slot reduction."""
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting

sys.path.insert(0, "/workspace/ATOM-tritonmoe/scripts")
from probe_moe_fixed_router import CAPTURE, load_weights
from probe_moe_slot_stage2 import remap_ids

OUT = Path(sys.argv[1])


def unshuffle(w):
    e, n, k = w.shape
    return w.view(e, n // 16, k // 32, 4, 16, 8).permute(0, 1, 4, 2, 3, 5).contiguous().view(e, n, k)


def metric(a, b):
    aa, bb = a.float(), b.float()
    delta = (aa - bb).abs()
    return {"shape": list(a.shape), "max": float(delta.max()),
            "mean": float(delta.mean()), "different": int((delta != 0).sum()),
            "total": delta.numel()}


def ordered_sum(per_slot):
    m, topk, dim = per_slot.shape
    total = torch.zeros((m, dim), dtype=torch.float32, device=per_slot.device)
    for j in range(topk):
        total.add_(per_slot[:, j, :].float())
    return total.to(torch.bfloat16)


def main():
    cap = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    if "--decode" in sys.argv:
        for key in ("h2", "moe", "sel", "rw"):
            cap[key] = cap[key][-1:]
    x = cap["h2"].cuda()
    ref = cap["moe"].cuda().to(torch.bfloat16)
    ids = cap["sel"].to(device="cuda", dtype=torch.int64)
    weights = cap["rw"].to(device="cuda", dtype=torch.bfloat16)
    assert torch.equal(cap["rw"], cap["rw"].to(torch.bfloat16).float())
    w13_ck, w2_ck = load_weights()
    w13_hf, w2_hf = unshuffle(w13_ck), unshuffle(w2_ck)
    m, topk = ids.shape
    num_experts, inter_dim, dim = len(w13_ck), w2_ck.shape[-1], x.shape[-1]

    # Installed Transformers 5.2.0 grouped_mm_experts_forward, lines 241-297.
    token_idx = torch.arange(m, device=x.device).unsqueeze(1).expand(-1, topk).reshape(-1)
    expert_ids = ids.reshape(-1)
    perm = torch.argsort(expert_ids)
    inv_perm = torch.argsort(perm)
    selected = x[token_idx][perm]
    sample_weights_g = weights.reshape(-1)[perm]
    sorted_expert_ids = expert_ids[perm]
    hist = torch.histc(sorted_expert_ids.int(), bins=num_experts, min=0, max=num_experts - 1)
    offs = torch.cumsum(hist, dim=0, dtype=torch.int32)
    gate_up = F.grouped_mm(selected, w13_hf.transpose(-2, -1), offs=offs)
    gate, up = gate_up.chunk(2, dim=-1)
    hf_gated_g = F.silu(gate) * up
    hf_down_g = F.grouped_mm(hf_gated_g, w2_hf.transpose(-2, -1), offs=offs)
    hf_term_g = hf_down_g * sample_weights_g.unsqueeze(-1)
    hf_gated = hf_gated_g[inv_perm].view(m, topk, inter_dim).contiguous()
    hf_down = hf_down_g[inv_perm].view(m, topk, dim).contiguous()
    hf_term = hf_term_g[inv_perm].view(m, topk, dim).contiguous()
    hf_final = hf_term.sum(dim=1)

    meta = get_2stage_cfgs(
        get_padded_M(m), dim, inter_dim, num_experts, topk,
        torch.bfloat16, torch.bfloat16, torch.bfloat16,
        QuantType.No, True, ActivationType.Silu, False, 0, 0, True,
    )
    assert not meta.run_1stage and meta.ksplit == 0
    sorted_ids, sorted_weights, sorted_experts, num_valid, _ = moe_sorting(
        ids.to(torch.int32), weights.float(), num_experts, dim,
        torch.bfloat16, meta.block_m,
    )
    torch.cuda.synchronize()
    flat_ids = remap_ids(sorted_ids, m, int(num_valid[0].item()))
    ck_gated = torch.empty((m, topk, inter_dim), dtype=torch.bfloat16, device=x.device)
    meta.stage1(
        x, w13_ck, w2_ck, sorted_ids, sorted_experts, num_valid, ck_gated, topk,
        block_m=meta.block_m, a1_scale=None, w1_scale=None, sorted_weights=None,
    )

    def debug_stage1(mode):
        os.environ["CK_MELLUM_STAGE1_MODE"] = str(mode)
        out = torch.empty_like(ck_gated)
        meta.stage1(x, w13_ck, w2_ck, sorted_ids, sorted_experts, num_valid,
                    out, topk, block_m=meta.block_m, a1_scale=None,
                    w1_scale=None, sorted_weights=None)
        torch.cuda.synchronize()
        return out

    ck_original = debug_stage1(0)
    ck_aligned = debug_stage1(1)
    ck_gate = debug_stage1(2)
    ck_up = debug_stage1(3)
    ck_silu = debug_stage1(4)
    os.environ["CK_MELLUM_STAGE1_MODE"] = "1"
    ck_gated = ck_aligned
    hf_gate = gate[inv_perm].view(m, topk, inter_dim).contiguous()
    hf_up = up[inv_perm].view(m, topk, inter_dim).contiguous()
    def fp64_mismatches(ck_out, hf_out, weight_offset):
        ix = torch.nonzero(ck_out != hf_out)
        if len(ix) == 0:
            return {"differing": 0}
        token, slot, feature = ix.unbind(1)
        expert = ids[token, slot]
        dot = (x[token].double() * w13_hf[expert, feature + weight_offset].double()).sum(-1)
        rounded = []
        midpoint = 0
        for v in dot.cpu().tolist():
            if v == 0:
                rounded.append(v)
                continue
            exponent = math.frexp(abs(v))[1] - 1
            scaled = math.ldexp(abs(v), 7 - exponent)
            midpoint += int(scaled - math.floor(scaled) == 0.5)
            rounded.append(math.copysign(math.ldexp(round(scaled), exponent - 7), v))
        r = torch.tensor(rounded, dtype=torch.float64, device=x.device)
        ck_match = ck_out[token, slot, feature].double() == r
        hf_match = hf_out[token, slot, feature].double() == r
        return {"differing": len(ix), "ck_matches_fp64_bf16": int(ck_match.sum()),
                "hf_matches_fp64_bf16": int(hf_match.sum()),
                "neither_matches_fp64_bf16": int((~ck_match & ~hf_match).sum()),
                "fp64_at_bf16_midpoint": midpoint}

    diagnostics = {
        "fp64_gate_mismatches": fp64_mismatches(ck_gate, hf_gate, 0),
        "fp64_up_mismatches": fp64_mismatches(ck_up, hf_up, inter_dim),
        "gemm_gate_ck_vs_hf": metric(ck_gate, hf_gate),
        "gemm_up_ck_vs_hf": metric(ck_up, hf_up),
        "silu_same_ck_gate_input": metric(ck_silu, F.silu(ck_gate)),
        "multiply_same_ck_silu_up_input": metric(ck_aligned, ck_silu * ck_up),
        "aligned_ck_vs_torch_ops_on_ck_gemm": metric(ck_aligned, F.silu(ck_gate) * ck_up),
        "original_stage1_vs_hf": metric(ck_original, hf_gated),
        "aligned_stage1_vs_hf": metric(ck_aligned, hf_gated),
    }

    def ck_stage2_per_slot(gated, use_weights=True, override_weights=None):
        out = torch.zeros((m * topk, dim), dtype=torch.bfloat16, device=x.device)
        meta.stage2(
            gated.view(m * topk, 1, inter_dim), w13_ck, w2_ck,
            flat_ids, sorted_experts, num_valid, out, 1,
            block_m=meta.block_m, w2_scale=None, a2_scale=None,
            sorted_weights=(override_weights if override_weights is not None
                            else sorted_weights) if use_weights else None,
        )
        return out.view(m, topk, dim)

    ck_term_ck_stage1 = ck_stage2_per_slot(ck_gated)
    ck_term_hf_stage1 = ck_stage2_per_slot(hf_gated)
    ck_down_ck_stage1 = ck_stage2_per_slot(ck_gated, use_weights=False)
    ck_down_ck_stage1_then_hf_weight = ck_down_ck_stage1 * weights.unsqueeze(-1)
    ck_down_hf_stage1 = ck_stage2_per_slot(hf_gated, use_weights=False)
    ck_down_then_hf_weight = ck_down_hf_stage1 * weights.unsqueeze(-1)
    ck_down_ones_hf_stage1 = ck_stage2_per_slot(
        hf_gated, override_weights=torch.ones_like(sorted_weights)
    )
    ck_down_ones_then_hf_weight = ck_down_ones_hf_stage1 * weights.unsqueeze(-1)
    torch.cuda.synchronize()
    result = {
        "stage1_diagnostics": diagnostics,
        "m": m, "topk": topk, "block_m": int(meta.block_m),
        "hf_grouped_reconstruction_vs_captured": metric(hf_final, ref),
        "ck_stage1_gated_vs_hf_grouped_gated": metric(ck_gated, hf_gated),
        "ck_stage2_terms_with_ck_stage1_vs_hf_terms": metric(ck_term_ck_stage1, hf_term),
        "ck_stage2_terms_with_hf_stage1_vs_hf_terms": metric(ck_term_hf_stage1, hf_term),
        "ck_stage2_unweighted_with_hf_stage1_vs_hf_down": metric(ck_down_hf_stage1, hf_down),
        "ck_stage2_weighted_kernel_with_ones_vs_hf_down": metric(ck_down_ones_hf_stage1, hf_down),
        "ck_stage2_weighted_kernel_ones_then_hf_weight_vs_hf_terms": metric(ck_down_ones_then_hf_weight, hf_term),
        "ck_stage2_unweighted_with_ck_stage1_vs_hf_down": metric(ck_down_ck_stage1, hf_down),
        "ck_stage2_ck_stage1_separate_weight_vs_hf_terms": metric(ck_down_ck_stage1_then_hf_weight, hf_term),
        "ck_stage2_unweighted_then_hf_weight_vs_hf_terms": metric(ck_down_then_hf_weight, hf_term),
        "ck_stage2_fused_weight_vs_separate_weight": metric(ck_term_hf_stage1, ck_down_then_hf_weight),
        "ck_stage2_ck_stage1_sum_vs_hf_final": metric(ck_term_ck_stage1.sum(dim=1), hf_final),
        "ck_stage2_hf_stage1_sum_vs_hf_final": metric(ck_term_hf_stage1.sum(dim=1), hf_final),
        "ck_down_then_hf_weight_sum_vs_hf_final": metric(ck_down_then_hf_weight.sum(dim=1), hf_final),
        "ck_down_ones_then_hf_weight_sum_vs_hf_final": metric(ck_down_ones_then_hf_weight.sum(dim=1), hf_final),
        "ck_down_ck_stage1_then_hf_weight_sum_vs_hf_final": metric(ck_down_ck_stage1_then_hf_weight.sum(dim=1), hf_final),
        "ck_stage2_ck_stage1_ordered_vs_hf_final": metric(ordered_sum(ck_term_ck_stage1), hf_final),
        "hf_terms_ordered_vs_hf_final": metric(ordered_sum(hf_term), hf_final),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
