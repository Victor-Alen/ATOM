#!/usr/bin/env python3
"""Run the same CK stage2 GEMM into one output row per original top-k slot.

The CK sorting ID packs token in low 24 bits and slot in high 8 bits. Remap
that logical ID to token * 8 + slot, keep sorted block order and weights, and
call the *same* selected stage2 function/block_m with topk=1 and flattened
stage1 a2. This changes only input/output row addressing. Each output row now
has one contributor; ordered reduction of those rows isolates combine error.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/workspace/ATOM-tritonmoe/scripts")
from probe_moe_fixed_router import CAPTURE, load_weights

REPEATS = 40
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/workspace/logs/step0_continuation/slot_stage2.json"
)


def digest(x: torch.Tensor) -> str:
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def error(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).abs()
    return {"max": float(d.max()), "mean": float(d.mean())}


def remap_ids(sorted_ids: torch.Tensor, m: int, valid_end: int) -> torch.Tensor:
    original = sorted_ids.cpu()
    token = original & 0xFFFFFF
    slot = (original >> 24) & 0xFF
    mask = token < m
    in_range = mask[:valid_end]
    mapped = (token[:valid_end][in_range].to(torch.int64) * 8
              + slot[:valid_end][in_range]).tolist()
    assert len(mapped) == m * 8, (len(mapped), m * 8)
    assert len(set(mapped)) == m * 8
    assert max(mapped) == m * 8 - 1
    assert bool((slot[:valid_end][in_range] < 8).all())
    # Padding IDs are ignored by CK but retain a sentinel outside output rows.
    result = torch.where(mask, token * 8 + slot, torch.full_like(token, m * 8))
    return result.to(device=sorted_ids.device, dtype=torch.int32)


def case(name: str, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
         ref: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor) -> dict:
    x = x.cuda()
    ids = ids.to(device="cuda", dtype=torch.int32)
    weights = weights.to(device="cuda", dtype=torch.float32)
    m = len(x)
    meta = get_2stage_cfgs(
        get_padded_M(m), 2304, 896, 64, 8,
        torch.bfloat16, torch.bfloat16, torch.bfloat16,
        QuantType.No, True, ActivationType.Silu, False, 0, 0, True,
    )
    assert not meta.run_1stage and meta.ksplit == 0
    sorted_ids, sorted_weights, sorted_experts, num_valid, _ = moe_sorting(
        ids, weights, 64, 2304, torch.bfloat16, meta.block_m
    )
    torch.cuda.synchronize()
    valid_end = int(num_valid[0].item())
    flat_ids = remap_ids(sorted_ids, m, valid_end)
    a2 = torch.empty((m, 8, 896), dtype=torch.bfloat16, device="cuda")
    meta.stage1(
        x, w13, w2, sorted_ids, sorted_experts, num_valid, a2, 8,
        block_m=meta.block_m, a1_scale=None, w1_scale=None, sorted_weights=None,
    )
    torch.cuda.synchronize()
    a2 = a2.clone()
    flat_a2 = a2.view(m * 8, 1, 896)
    ref = ref.float().cpu()
    atomic_hashes, slot_hashes, ordered_hashes = [], [], []
    first_atomic = first_ordered = first_ordered32 = None
    for _ in range(REPEATS):
        atomic = torch.zeros((m, 2304), dtype=torch.bfloat16, device="cuda")
        meta.stage2(
            a2, w13, w2, sorted_ids, sorted_experts, num_valid, atomic, 8,
            block_m=meta.block_m, w2_scale=None, a2_scale=None,
            sorted_weights=sorted_weights,
        )
        per_slot = torch.zeros((m * 8, 2304), dtype=torch.bfloat16, device="cuda")
        meta.stage2(
            flat_a2, w13, w2, flat_ids, sorted_experts, num_valid, per_slot, 1,
            block_m=meta.block_m, w2_scale=None, a2_scale=None,
            sorted_weights=sorted_weights,
        )
        per_slot = per_slot.view(m, 8, 2304)
        ordered32 = torch.zeros((m, 2304), dtype=torch.float32, device="cuda")
        for j in range(8):
            ordered32.add_(per_slot[:, j, :].float())
        ordered = ordered32.to(torch.bfloat16)
        torch.cuda.synchronize()
        atomic_cpu, slot_cpu, ordered_cpu = atomic.cpu(), per_slot.cpu(), ordered.cpu()
        atomic_hashes.append(digest(atomic_cpu))
        slot_hashes.append(digest(slot_cpu))
        ordered_hashes.append(digest(ordered_cpu))
        if first_atomic is None:
            first_atomic = atomic_cpu
            first_ordered = ordered_cpu
            first_ordered32 = ordered32.cpu()
    profile_atomic = torch.zeros((m, 2304), dtype=torch.bfloat16, device="cuda")
    profile_slots = torch.zeros((m * 8, 2304), dtype=torch.bfloat16, device="cuda")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        meta.stage2(
            a2, w13, w2, sorted_ids, sorted_experts, num_valid, profile_atomic, 8,
            block_m=meta.block_m, w2_scale=None, a2_scale=None,
            sorted_weights=sorted_weights,
        )
        meta.stage2(
            flat_a2, w13, w2, flat_ids, sorted_experts, num_valid,
            profile_slots, 1, block_m=meta.block_m, w2_scale=None,
            a2_scale=None, sorted_weights=sorted_weights,
        )
        torch.cuda.synchronize()
    stage2_kernel_names = [e.name for e in prof.events()
                           if e.device_type == torch.autograd.DeviceType.CUDA
                           and "kernel_moe_gemm" in e.name]
    assert len(stage2_kernel_names) == 2, stage2_kernel_names
    result = {
        "case": name, "m": m, "block_m": int(meta.block_m),
        "stage2_function": getattr(meta.stage2, "__name__", str(meta.stage2)),
        "stage1_shape": list(a2.shape), "per_slot_shape": [m, 8, 2304],
        "valid_sorted_ids": valid_end, "mapped_slots": m * 8,
        "repeats": REPEATS,
        "distinct_atomic": len(set(atomic_hashes)),
        "distinct_per_slot": len(set(slot_hashes)),
        "distinct_ordered": len(set(ordered_hashes)),
        "same_profiled_stage2_kernel_name": stage2_kernel_names[0] == stage2_kernel_names[1],
        "profiled_stage2_kernel_name_sha256": [hashlib.sha256(n.encode()).hexdigest()[:16]
                                               for n in stage2_kernel_names],
        "atomic_vs_hf": error(first_atomic, ref),
        "ordered_bf16_vs_hf": error(first_ordered, ref),
        "ordered_fp32_vs_hf": error(first_ordered32, ref),
        "atomic_vs_ordered_bf16": error(first_atomic, first_ordered),
        "atomic_hashes": atomic_hashes,
        "ordered_hashes": ordered_hashes,
    }
    print(json.dumps({k: v for k, v in result.items()
                      if not k.endswith("_hashes")}), flush=True)
    return result


def main() -> None:
    captured = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    w13, w2 = load_weights()
    results = []
    for name, sl in (("decode_1", slice(-1, None)), ("prefill_866", slice(None))):
        results.append(case(name, captured["h2"][sl], captured["sel"][sl],
                            captured["rw"][sl], captured["moe"][sl], w13, w2))
    OUT.write_text(json.dumps({"visible_devices": os.getenv("HIP_VISIBLE_DEVICES"),
                               "results": results}, indent=2) + "\n")
    print("OUTPUT", OUT, flush=True)


if __name__ == "__main__":
    main()
