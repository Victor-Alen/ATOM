#!/usr/bin/env python3
"""Isolate Mellum's AITER MoE from routing using real layer-0 data and weights."""
import collections
import hashlib
import json
import sys
from pathlib import Path

import aiter
import torch
from aiter import ActivationType
from aiter import QuantType
from aiter.fused_moe import fused_moe, get_2stage_cfgs, get_padded_M, moe_sorting
from aiter.ops.shuffle import shuffle_weight
from safetensors import safe_open

MODEL = Path("/data/models/Mellum2.1-12B-A2.5B-Thinking")
CAPTURE = Path("/workspace/logs/hf_moe_l0.pt")
WEIGHT_CHECK = Path("/workspace/logs/prodw.pt")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/workspace/logs/step0_continuation/fixed_router.json"
)
REPEATS = 40


def digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def load_weights():
    path = MODEL / "model-00001-of-00005.safetensors"
    gate_up, down = [], []
    with safe_open(path, framework="pt", device="cpu") as f:
        for expert in range(64):
            stem = f"model.layers.0.mlp.experts.{expert}."
            gate = f.get_tensor(stem + "gate_proj.weight")
            up = f.get_tensor(stem + "up_proj.weight")
            gate_up.append(torch.cat((gate, up), dim=0))
            down.append(f.get_tensor(stem + "down_proj.weight"))
    w13 = shuffle_weight(torch.stack(gate_up).to("cuda:0"))
    w2 = shuffle_weight(torch.stack(down).to("cuda:0"))
    check = torch.load(WEIGHT_CHECK, map_location="cpu", weights_only=False)
    assert torch.equal(w13[0].cpu(), check["w13_e0"].to(w13.dtype))
    assert torch.equal(w2[0].cpu(), check["w2_e0"].to(w2.dtype))
    return w13, w2


def direct_moe(x, w13, w2, weights, ids):
    return fused_moe(
        hidden_states=x,
        w1=w13,
        w2=w2,
        topk_weight=weights,
        topk_ids=ids,
        activation=ActivationType.Silu,
    )


def probe_case(name, x, ref, ids, weights, w13, w2):
    x, ref = x.to("cuda:0"), ref.to("cuda:0")
    ids = ids.to(device="cuda:0", dtype=torch.int32)
    weights = weights.to(device="cuda:0", dtype=torch.float32)
    ids_before, weights_before = ids.clone(), weights.clone()
    outputs = []
    for _ in range(REPEATS):
        value = direct_moe(x, w13, w2, weights, ids)
        torch.cuda.synchronize()
        outputs.append(value.cpu())
    assert torch.equal(ids, ids_before) and torch.equal(weights, weights_before)
    hashes = [digest(x) for x in outputs]
    first = outputs[0].float()
    repeat_deltas = [(value.float() - first).abs() for value in outputs[1:]]
    ref_delta = (first - ref.cpu().float()).abs()
    result = {
        "case": name,
        "tokens": len(x),
        "repeats": REPEATS,
        "distinct_output_hashes": len(set(hashes)),
        "hash_counts": dict(collections.Counter(hashes)),
        "max_repeat_delta": max(float(x.max()) for x in repeat_deltas),
        "changed_elements": max(int((x > 0).sum()) for x in repeat_deltas),
        "vs_hf_max": float(ref_delta.max()),
        "vs_hf_mean": float(ref_delta.mean()),
        "routing_unchanged": True,
    }
    print(json.dumps(result), flush=True)
    return result


def probe_routing(name, logits):
    logits = logits.to("cuda:0")
    hashes = []
    first_ids = first_weights = None
    for repeat in range(REPEATS):
        ids = torch.empty((len(logits), 8), dtype=torch.int32, device=logits.device)
        weights = torch.empty((len(logits), 8), dtype=torch.float32, device=logits.device)
        token_expert = torch.empty_like(ids)
        aiter.topk_softmax(weights, ids, token_expert, logits, True, 0, "")
        torch.cuda.synchronize()
        ids, weights = ids.cpu(), weights.cpu()
        if repeat == 0:
            first_ids, first_weights = ids, weights
        hashes.append(digest(ids) + digest(weights))
    result = {
        "case": name,
        "repeats": REPEATS,
        "distinct_routing_hashes": len(set(hashes)),
        "ids_equal_first": all(h == hashes[0] for h in hashes),
    }
    print(json.dumps(result), flush=True)
    return result


def probe_stages(name, x, ids, weights, w13, w2):
    """Freeze sorting and stage1, then repeat only the CK stage2 combine."""
    x = x.to("cuda:0")
    ids = ids.to(device="cuda:0", dtype=torch.int32)
    weights = weights.to(device="cuda:0", dtype=torch.float32)
    meta = get_2stage_cfgs(
        get_padded_M(len(x)), 2304, 896, 64, 8,
        torch.bfloat16, torch.bfloat16, torch.bfloat16,
        QuantType.No, True, ActivationType.Silu, False, 0, 0, True,
    )
    assert not meta.run_1stage and meta.ksplit == 0
    sorted_ids, sorted_weights, sorted_experts, num_valid, _ = moe_sorting(
        ids, weights, 64, 2304, torch.bfloat16, meta.block_m
    )
    inter = torch.empty((len(x), 8, 896), dtype=torch.bfloat16, device=x.device)
    meta.stage1(
        x, w13, w2, sorted_ids, sorted_experts, num_valid, inter, 8,
        block_m=meta.block_m, a1_scale=None, w1_scale=None, sorted_weights=None,
    )
    torch.cuda.synchronize()
    inter_fixed = inter.clone()

    stage1_hashes = []
    for _ in range(REPEATS):
        tmp = torch.empty_like(inter)
        meta.stage1(
            x, w13, w2, sorted_ids, sorted_experts, num_valid, tmp, 8,
            block_m=meta.block_m, a1_scale=None, w1_scale=None, sorted_weights=None,
        )
        torch.cuda.synchronize()
        stage1_hashes.append(digest(tmp.cpu()))

    stage2_outputs = []
    for _ in range(REPEATS):
        out = torch.zeros((len(x), 2304), dtype=torch.bfloat16, device=x.device)
        meta.stage2(
            inter_fixed, w13, w2, sorted_ids, sorted_experts, num_valid, out, 8,
            block_m=meta.block_m, w2_scale=None, a2_scale=None,
            sorted_weights=sorted_weights,
        )
        torch.cuda.synchronize()
        stage2_outputs.append(out.cpu())
    first = stage2_outputs[0].float()
    deltas = [(item.float() - first).abs() for item in stage2_outputs[1:]]
    result = {
        "case": "frozen_stages_" + name,
        "repeats": REPEATS,
        "distinct_stage1_hashes": len(set(stage1_hashes)),
        "distinct_stage2_hashes": len(set(digest(x) for x in stage2_outputs)),
        "max_stage2_delta": max(float(x.max()) for x in deltas),
        "changed_stage2_elements": max(int((x > 0).sum()) for x in deltas),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    captured = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    gate = torch.load("/workspace/logs/gate_engine.pt", map_location="cpu", weights_only=False)
    assert torch.equal(captured["sel"], gate["topi"])
    assert torch.equal(captured["rw"], gate["topv"])
    w13, w2 = load_weights()
    results = []
    for name, sl in (("decode_1", slice(-1, None)), ("prefill_866", slice(None))):
        results.append(probe_routing("routing_" + name, gate["gate_logits"][sl]))
        results.append(probe_case(
            name,
            captured["h2"][sl],
            captured["moe"][sl],
            captured["sel"][sl],
            captured["rw"][sl],
            w13,
            w2,
        ))
        results.append(probe_stages(
            name, captured["h2"][sl], captured["sel"][sl],
            captured["rw"][sl], w13, w2,
        ))
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print("OUTPUT", OUT, flush=True)


if __name__ == "__main__":
    main()
