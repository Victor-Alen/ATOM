#!/usr/bin/env python3
"""Latency comparison: production AITER MoE vs same-CK-stage2 ordered slots.

Uses captured Mellum L0 inputs/weights/routes. Times one complete MoE call
(sorting + stage1 + stage2, plus remap/reduce for ordered mode), excluding gate
GEMM and routing. Timings include Python launch overhead and synchronization.
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from aiter import ActivationType, QuantType
from aiter.fused_moe import get_2stage_cfgs, get_padded_M, moe_sorting

sys.path.insert(0, "/workspace/ATOM-tritonmoe/scripts")
from probe_moe_fixed_router import CAPTURE, direct_moe, load_weights

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/workspace/logs/step0_continuation/bench_slot_stage2.json"
)
WARMUP = 10
REPEATS = 50


def digest(x):
    return hashlib.sha256(x.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def ordered_moe(x, w13, w2, weights, ids):
    m, topk = ids.shape
    e, model_dim, inter_dim = len(w13), w2.shape[1], w2.shape[2]
    meta = get_2stage_cfgs(
        get_padded_M(m), model_dim, inter_dim, e, topk,
        x.dtype, w13.dtype, w13.dtype, QuantType.No,
        inter_dim != w13.shape[1], ActivationType.Silu,
        False, 0, 0,
        getattr(w13, "is_shuffled", False) or getattr(w2, "is_shuffled", False),
    )
    assert not meta.run_1stage and meta.ksplit == 0
    sorted_ids, sorted_weights, sorted_experts, num_valid, _ = moe_sorting(
        ids, weights, e, model_dim, x.dtype, meta.block_m
    )
    a2 = torch.empty((m, topk, inter_dim), dtype=x.dtype, device=x.device)
    meta.stage1(
        x, w13, w2, sorted_ids, sorted_experts, num_valid, a2, topk,
        block_m=meta.block_m, a1_scale=None, w1_scale=None, sorted_weights=None,
    )
    token = sorted_ids.bitwise_and(0xFFFFFF)
    slot = sorted_ids.bitwise_right_shift(24).bitwise_and(0xFF)
    flat_ids = torch.where(token < m, token * topk + slot,
                           torch.full_like(token, m * topk))
    per_slot = torch.zeros((m * topk, model_dim), dtype=x.dtype, device=x.device)
    meta.stage2(
        a2.view(m * topk, 1, inter_dim), w13, w2, flat_ids,
        sorted_experts, num_valid, per_slot, 1,
        block_m=meta.block_m, w2_scale=None, a2_scale=None,
        sorted_weights=sorted_weights,
    )
    per_slot = per_slot.view(m, topk, model_dim)
    total = torch.zeros((m, model_dim), dtype=torch.float32, device=x.device)
    for slot_id in range(topk):
        total.add_(per_slot[:, slot_id, :].float())
    return total.to(x.dtype)


def summarize(samples):
    values = sorted(samples)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p10_ms": values[int((len(values)-1)*0.1)],
        "p90_ms": values[int((len(values)-1)*0.9)],
        "min_ms": values[0], "max_ms": values[-1],
    }


def bench_case(name, x, ids, weights, w13, w2):
    x = x.cuda()
    ids = ids.to(device="cuda", dtype=torch.int32)
    weights = weights.to(device="cuda", dtype=torch.float32)
    calls = {
        "production_fused": lambda: direct_moe(x, w13, w2, weights, ids),
        "ordered_slots": lambda: ordered_moe(x, w13, w2, weights, ids),
    }
    for _ in range(WARMUP):
        for fn in calls.values():
            fn()
    torch.cuda.synchronize()
    hashes = {label: digest(fn()) for label, fn in calls.items()}
    samples = {label: [] for label in calls}
    for i in range(REPEATS):
        order = tuple(calls) if i % 2 == 0 else tuple(reversed(calls))
        for label in order:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            calls[label]()
            torch.cuda.synchronize()
            samples[label].append((time.perf_counter_ns() - start) / 1e6)
    baseline = summarize(samples["production_fused"])
    ordered = summarize(samples["ordered_slots"])
    result = {
        "case": name, "m": len(x), "warmup": WARMUP,
        "repeats_per_mode": REPEATS,
        "unit": "synchronized complete MoE call, gate/routing excluded",
        "hashes": hashes,
        "production_fused": baseline,
        "ordered_slots": ordered,
        "median_slowdown_x": ordered["median_ms"] / baseline["median_ms"],
        "raw_samples_ms": samples,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "raw_samples_ms"}), flush=True)
    return result


def main():
    captured = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    w13, w2 = load_weights()
    results = []
    for name, sl in (("decode_1", slice(-1, None)), ("prefill_866", slice(None))):
        results.append(bench_case(name, captured["h2"][sl], captured["sel"][sl],
                                  captured["rw"][sl], w13, w2))
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print("OUTPUT", OUT, flush=True)


if __name__ == "__main__":
    main()
