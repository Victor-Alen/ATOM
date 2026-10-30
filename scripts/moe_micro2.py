#!/usr/bin/env python3
"""MoE micro-test v2.

Fixes the by-reference hook capture (clone) and adds the ground-truth direct module
call, so we can tell "my loop is wrong" apart from "the captured tensor was mutated".
"""
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REFERENCE = Path("/workspace/ATOM/scripts/hf_reference_generate.py")


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def my_moe(layer, cfg, h):
    routed = h.view(-1, cfg.hidden_size)
    logits_r = F.linear(routed, layer.mlp.gate.weight)
    probs_r = F.softmax(logits_r, dtype=torch.float, dim=-1)
    topv, topi = torch.topk(probs_r, cfg.num_experts_per_tok, dim=-1)
    if cfg.norm_topk_prob:
        topv = topv / topv.sum(dim=-1, keepdim=True)
    topv = topv.to(logits_r.dtype)
    final = torch.zeros_like(routed)
    em = F.one_hot(topi, num_classes=cfg.num_experts + 1).permute(2, 1, 0)
    gu = layer.mlp.experts.gate_up_proj
    dn = layer.mlp.experts.down_proj
    for e in range(cfg.num_experts):
        if int(em[e].sum()) == 0:
            continue
        si, ti = torch.where(em[e])
        cur = routed[ti]
        gate_up = F.linear(cur, gu[e])
        g, u = gate_up.chunk(2, dim=-1)
        outp = F.linear(F.silu(g) * u, dn[e]) * topv[ti, si, None]
        final.index_add_(0, ti, outp.to(final.dtype))
    return final.view_as(h)


def main():
    ref = load_reference_module()
    MellumConfig, MellumForCausalLM = ref.load_mellum_classes()
    cfg = MellumConfig.from_pretrained(MODEL_PATH)
    cfg._attn_implementation = "eager"
    print("loading HF ...", flush=True)
    model = MellumForCausalLM.from_pretrained(
        MODEL_PATH, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    ref.load_checkpoint_experts(model, MODEL_PATH)
    model = model.to("cuda:1").eval()
    print("HF ready", flush=True)

    rec = json.loads(open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8").readline())
    ids = torch.tensor([rec["prompt_token_ids"][:64]], dtype=torch.long, device="cuda:1")
    layer = model.model.layers[0]
    cap = {}

    def hook(mod, inp, out):
        cap["in"] = inp[0].detach().clone()
        cap["out"] = (out[0] if isinstance(out, (tuple, list)) else out).detach().clone()

    handle = layer.mlp.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=ids, logits_to_keep=1)
    handle.remove()

    h = cap["in"]
    hooked = cap["out"]
    with torch.no_grad():
        direct = layer.mlp(h)
        mine = my_moe(layer, model.config, h)

    def rep(tag, a, b):
        d = (a.float() - b.float()).abs()
        print(f"{tag}: max|delta|={d.max().item():.8f} differing={int((d > 0).sum())}/{d.numel()}", flush=True)

    rep("module(h) vs hooked", direct, hooked)
    rep("mine vs hooked", mine, hooked)
    rep("mine vs module(h)", mine, direct)

    # rerun the model a second time and compare logits: is HF itself reproducible?
    with torch.no_grad():
        o1 = model(input_ids=ids, logits_to_keep=1).logits.clone()
        o2 = model(input_ids=ids, logits_to_keep=1).logits.clone()
        o3 = model(input_ids=ids, logits_to_keep=1).logits.clone()
    for i, o in enumerate((o2, o3), start=2):
        d = (o1.float() - o.float()).abs()
        print(f"HF run1 vs run{i}: max|delta|={d.max().item():.8f} differing={int((d > 0).sum())}/{d.numel()}", flush=True)
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
