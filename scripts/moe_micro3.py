#!/usr/bin/env python3
"""MoE micro-test v3: bisect the last 1-ulp difference.

(1) compares the router path (logits, probabilities, top-k values/indices),
(2) runs HF's expert loop copied verbatim,
(3) runs my compact loop,
and reports which one deviates from the module's own output.
"""
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REFERENCE = Path("/workspace/ATOM/scripts/hf_reference_generate.py")


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verbatim(experts, hidden_states, top_k_index, top_k_weights):
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
        current_hidden_states = experts.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, experts.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


def mine(experts, hidden_states, top_k_index, top_k_weights, use_mask=False):
    final = torch.zeros_like(hidden_states)
    if use_mask:
        em = F.one_hot(top_k_index, num_classes=experts.num_experts + 1).permute(2, 1, 0)
        for e in range(experts.num_experts):
            if int(em[e].sum()) == 0:
                continue
            si, ti = torch.where(em[e])
            cur = hidden_states[ti]
            g, u = F.linear(cur, experts.gate_up_proj[e]).chunk(2, dim=-1)
            o = F.linear(F.silu(g) * u, experts.down_proj[e]) * top_k_weights[ti, si, None]
            final.index_add_(0, ti, o.to(final.dtype))
    else:
        for e in range(experts.num_experts):
            m = top_k_index == e
            if not bool(m.any()):
                continue
            ti, si = torch.where(m)
            cur = hidden_states[ti]
            g, u = F.linear(cur, experts.gate_up_proj[e]).chunk(2, dim=-1)
            o = F.linear(F.silu(g) * u, experts.down_proj[e]) * top_k_weights[ti, si, None]
            final.index_add_(0, ti, o.to(final.dtype))
    return final


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

    def hook_mlp(mod, inp, out):
        cap["mlp_in"] = inp[0].detach().clone()
        cap["mlp_out"] = (out[0] if isinstance(out, (tuple, list)) else out).detach().clone()

    def hook_gate(mod, inp, out):
        cap["gate_in"] = inp[0].detach().clone()
        cap["gate_out"] = [t.detach().clone() for t in out]

    h1 = layer.mlp.register_forward_hook(hook_mlp)
    h2 = layer.mlp.gate.register_forward_hook(hook_gate)
    with torch.no_grad():
        model(input_ids=ids, logits_to_keep=1)
    h1.remove()
    h2.remove()

    gate_w = layer.mlp.gate.weight
    hs = cap["mlp_in"].view(-1, cfg.hidden_size)
    logits_r = F.linear(hs, gate_w)
    probs = F.softmax(logits_r, dtype=torch.float, dim=-1)
    topv, topi = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
    if cfg.norm_topk_prob:
        topv = topv / topv.sum(dim=-1, keepdim=True)
    topv_bf = topv.to(logits_r.dtype)

    hv_logits, hv_scores, hv_idx = cap["gate_out"]
    print(f"router logits equal: {torch.equal(logits_r, hv_logits)}  max|d|={(logits_r.float()-hv_logits.float()).abs().max().item():.8f}", flush=True)
    print(f"router idx equal: {torch.equal(topi, hv_idx)}", flush=True)
    print(f"router scores equal: {torch.equal(topv_bf, hv_scores)} max|d|={(topv_bf.float()-hv_scores.float()).abs().max().item():.8f}", flush=True)

    with torch.no_grad():
        vs = verbatim(layer.mlp.experts, hs, topi, topv_bf)
        m1 = mine(layer.mlp.experts, hs, topi, topv_bf, use_mask=True)
        m2 = mine(layer.mlp.experts, hs, topi, topv_bf, use_mask=False)
        ref_out = cap["mlp_out"].view_as(vs)
    for tag, t in (("verbatim", vs), ("mine(mask-order)", m1), ("mine(token-order)", m2)):
        d = (t.float() - ref_out.float()).abs()
        print(f"{tag} vs module: max|d|={d.max().item():.8f} differing={int((d>0).sum())}/{d.numel()}", flush=True)
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
