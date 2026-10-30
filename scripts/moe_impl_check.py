#!/usr/bin/env python3
"""Which MoE implementation does HF actually run, and does the eager loop match it?"""
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
    gu = layer.mlp.experts.gate_up_proj
    dn = layer.mlp.experts.down_proj
    for e in range(cfg.num_experts):
        m = topi == e
        if not bool(m.any()):
            continue
        ti, si = torch.where(m)
        cur = routed[ti]
        gate_up = F.linear(cur, gu[e])
        g, u = gate_up.chunk(2, dim=-1)
        outp = F.linear(F.silu(g) * u, dn[e]) * topv[ti, si, None]
        final.index_add_(0, ti, outp.to(final.dtype))
    return final.view_as(h)


def main():
    ref = load_reference_module()
    MellumConfig, MellumForCausalLM = ref.load_mellum_classes()
    config = MellumConfig.from_pretrained(MODEL_PATH)

    for impl in (None, "eager"):
        cfg = MellumConfig.from_pretrained(MODEL_PATH)
        cfg._attn_implementation = "eager"
        if impl is not None:
            cfg.experts_implementation = impl
        print(f"--- experts_implementation={impl} ---", flush=True)
        model = MellumForCausalLM.from_pretrained(
            MODEL_PATH, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        ref.load_checkpoint_experts(model, MODEL_PATH)
        model = model.to("cuda:1").eval()
        experts_mod = model.model.layers[0].mlp.experts
        print("resolved impl attr:", getattr(experts_mod, "_hf_experts_implementation", "n/a") or "n/a", flush=True)
        print("forward is:", getattr(experts_mod.forward, "__qualname__", "?"), flush=True)

        rec = json.loads(open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8").readline())
        ids = torch.tensor([rec["prompt_token_ids"][:64]], dtype=torch.long, device="cuda:1")
        layer = model.model.layers[0]
        cap = {}

        def hook(mod, inp, out):
            cap["in"] = inp[0].detach()
            cap["out"] = out.detach() if not isinstance(out, (tuple, list)) else out[0].detach()

        h = layer.mlp.register_forward_hook(hook)
        with torch.no_grad():
            model(input_ids=ids, logits_to_keep=1)
        h.remove()
        mine = my_moe(layer, model.config, cap["in"])
        d = (mine.float() - cap["out"].float()).abs()
        print(f"eager-loop vs HF: max|delta|={d.max().item():.8f} differing={int((d > 0).sum())}/{d.numel()}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
