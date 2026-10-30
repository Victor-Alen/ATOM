#!/usr/bin/env python3
"""Which MoE implementation does HF dispatch to, and does it change with T?

Compares the module's own output with (a) HF's batched_mm implementation and
(b) HF's eager loop, at T=64 and T=2048.
"""
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.integrations.moe import batched_mm_experts_forward

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REFERENCE = Path("/workspace/ATOM/scripts/hf_reference_generate.py")


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eager_loop(experts, hidden_states, top_k_index, top_k_weights):
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=experts.num_experts + 1)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == experts.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        h = experts.act_fn(gate) * up
        h = nn.functional.linear(h, experts.down_proj[expert_idx])
        h = h * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
    return final_hidden_states


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
    print("HF ready; config._experts_implementation =", getattr(model.config, "_experts_implementation", "MISSING"), flush=True)

    rec = json.loads(open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8").readline())
    layer = model.model.layers[0]
    experts = layer.mlp.experts

    for T in (64, 512, 2048):
        ids = torch.tensor([rec["prompt_token_ids"][:T]], dtype=torch.long, device="cuda:1")
        cap = {}

        def hook(mod, inp, out):
            cap["in"] = inp[0].detach().clone()
            cap["out"] = (out[0] if isinstance(out, (tuple, list)) else out).detach().clone()

        h = layer.mlp.register_forward_hook(hook)
        with torch.no_grad():
            model(input_ids=ids, logits_to_keep=1)
        h.remove()
        hs = cap["in"].view(-1, model.config.hidden_size)
        gw = layer.mlp.gate.weight
        logits_r = F.linear(hs, gw)
        probs = F.softmax(logits_r, dtype=torch.float, dim=-1)
        topv, topi = torch.topk(probs, model.config.num_experts_per_tok, dim=-1)
        if model.config.norm_topk_prob:
            topv = topv / topv.sum(dim=-1, keepdim=True)
        topv = topv.to(logits_r.dtype)

        ref_out = cap["out"].view_as(hs)
        with torch.no_grad():
            try:
                b = batched_mm_experts_forward(experts, hs, topi, topv)
                db = (b.float() - ref_out.float()).abs()
                rb = f"max|d|={db.max().item():.8f} diff={int((db>0).sum())}"
            except Exception as exc:  # noqa: BLE001
                rb = f"batched_mm failed: {type(exc).__name__}: {str(exc)[:90]}"
            e = eager_loop(experts, hs, topi, topv)
            de = (e.float() - ref_out.float()).abs()
            re_ = f"max|d|={de.max().item():.8f} diff={int((de>0).sum())}"
        print(f"T={T}: batched_mm vs module -> {rb} | eager vs module -> {re_}", flush=True)
        del cap, b, e, ref_out
        torch.cuda.empty_cache()
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
