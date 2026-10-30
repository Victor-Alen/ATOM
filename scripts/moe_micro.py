#!/usr/bin/env python3
"""Micro-test: run HF's own layer-0 MoE on its own captured input and compare with
my independent MoE implementation on the exact same tensor."""
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REFERENCE = Path("/workspace/ATOM/scripts/hf_reference_generate.py")


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ref = load_reference_module()
    MellumConfig, MellumForCausalLM = ref.load_mellum_classes()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = MellumConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "eager"
    print("loading HF ...", flush=True)
    model = MellumForCausalLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    ref.load_checkpoint_experts(model, MODEL_PATH)
    model = model.to("cuda:1").eval()
    print("HF ready", flush=True)

    rec = json.loads(open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8").readline())
    ids = torch.tensor([rec["prompt_token_ids"][:64]], dtype=torch.long, device="cuda:1")

    layer = model.model.layers[0]
    captured = {}

    def hook(mod, inp, out):
        captured["in"] = inp[0].detach()
        captured["out"] = out.detach() if not isinstance(out, (tuple, list)) else out[0].detach()

    handle = layer.mlp.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=ids, logits_to_keep=1)
    handle.remove()

    h = captured["in"]
    hf_out = captured["out"]
    print("captured input", tuple(h.shape), "hf out", tuple(hf_out.shape), flush=True)

    cfg = model.config
    routed = h.view(-1, cfg.hidden_size)
    logits_r = F.linear(routed, layer.mlp.gate.weight)
    probs_r = F.softmax(logits_r, dtype=torch.float, dim=-1)
    topv, topi = torch.topk(probs_r, cfg.num_experts_per_tok, dim=-1)
    if cfg.norm_topk_prob:
        topv = topv / topv.sum(dim=-1, keepdim=True)
    topv = topv.to(logits_r.dtype)

    # variant A: expert loop, accumulation in bf16, ascending expert id
    final = torch.zeros_like(routed)
    gu = layer.mlp.experts.gate_up_proj
    dn = layer.mlp.experts.down_proj
    with torch.no_grad():
        em = F.one_hot(topi, num_classes=cfg.num_experts + 1).permute(2, 1, 0)
        for e in range(cfg.num_experts):
            if int(em[e].sum()) == 0:
                continue
            si, ti = torch.where(em[e])
            cur = routed[ti]
            gate_up = F.linear(cur, gu[e])
            g, u = gate_up.chunk(2, dim=-1)
            inter = F.silu(g) * u
            outp = F.linear(inter, dn[e]) * topv[ti, si, None]
            final.index_add_(0, ti, outp.to(final.dtype))

    d = (final.view_as(hf_out).float() - hf_out.float()).abs()
    print(f"my MoE vs HF MoE on identical input: max|delta| = {d.max().item():.8f}")
    print(f"num differing elements: {int((d > 0).sum())}/{d.numel()}")
    if d.max().item() > 0:
        idx = int(d.argmax())
        print("worst element idx", idx, "mine", final.view(-1)[idx].item(), "hf", hf_out.view(-1)[idx].item())
        # retry with HF's exact expert ordering but router weights recomputed
        logits_r2 = F.linear(routed, layer.mlp.gate.weight)
        probs2 = F.softmax(logits_r2, dtype=torch.float, dim=-1)
        t2, i2 = torch.topk(probs2, cfg.num_experts_per_tok, dim=-1)
        print("router identical:", bool(torch.equal(topi, i2)), "weights equal:", bool(torch.equal(topv, t2.to(logits_r2.dtype))))
        # variant B: same loop but accumulate in fp32 then cast
        final32 = torch.zeros(routed.shape, dtype=torch.float32, device=routed.device)
        for e in range(cfg.num_experts):
            m = topi == e
            if not bool(m.any()):
                continue
            ti, si = torch.where(m)
            cur = routed[ti]
            gate_up = F.linear(cur, gu[e])
            g, u = gate_up.chunk(2, dim=-1)
            outp = F.linear(F.silu(g) * u, dn[e]) * topv[ti, si, None]
            final32.index_add_(0, ti, outp.float())
        d32 = (final32.view_as(hf_out).float() - hf_out.float()).abs()
        print(f"variant B (fp32 accumulate): max|delta| = {d32.max().item():.8f}")
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
