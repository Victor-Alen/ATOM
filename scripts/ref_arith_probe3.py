#!/usr/bin/env python3
"""v3: per-layer comparison between HF and an independent forward.

Uses HF's own rotary module for cos/sin (so the rope variant, including any YaRN
configuration, cannot be the source of disagreement) and reports the first layer
where my forward diverges from HF's hooked output.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.integrations.moe import grouped_mm_experts_forward

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REFERENCE = Path("/workspace/ATOM/scripts/hf_reference_generate.py")


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hf_norm(x, weight, eps):
    f = x.float()
    var = f.pow(2).mean(-1, keepdim=True)
    n = (f * torch.rsqrt(var + eps)).to(x.dtype)
    return weight * n


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-id", default="C-0000")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--window-rule", default="lt", choices=["lt", "le", "none"])
    args = ap.parse_args()

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
    model = model.to(args.device).eval()
    print("HF ready", flush=True)

    rec = None
    for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["id"] == args.prompt_id:
            rec = r
            break
    prompt_ids = rec["prompt_token_ids"][: args.prompt_tokens]
    ref_tokens = rec["token_ids"][: args.steps]
    ids = torch.tensor([prompt_ids + ref_tokens], dtype=torch.long, device=args.device)
    P = len(prompt_ids)
    T = ids.shape[1]

    hooks = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def mk(i, tag):
            def fn(mod, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                hooks[(i, tag)] = t.detach()
            return fn
        handles.append(layer.self_attn.register_forward_hook(mk(i, "attn")))
        handles.append(layer.mlp.register_forward_hook(mk(i, "mlp")))
        handles.append(layer.register_forward_hook(mk(i, "block")))
    with torch.no_grad():
        out = model(input_ids=ids, logits_to_keep=int(T))
    hf_logits = out.logits[0].float()
    for h in handles:
        h.remove()
    del out
    torch.cuda.empty_cache()
    print("HF done", flush=True)

    cfg = model.config
    dev = args.device
    eps = cfg.rms_norm_eps
    nH, nKV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    scaling = D ** -0.5
    W = cfg.sliding_window
    layers = model.model.layers
    pos = torch.arange(T, device=dev)
    j = torch.arange(T, device=dev)
    causal = j[None, :] <= pos[:, None]

    first_bad = None
    with torch.no_grad():
        x = F.embedding(ids, model.model.embed_tokens.weight)
        for li, layer in enumerate(layers):
            attn = layer.self_attn
            h = hf_norm(x, layer.input_layernorm.weight, eps)
            q = F.linear(h, attn.q_proj.weight).view(1, T, nH, D)
            k = F.linear(h, attn.k_proj.weight).view(1, T, nKV, D)
            v = F.linear(h, attn.v_proj.weight).view(1, T, nKV, D)
            q = hf_norm(q, attn.q_norm.weight, getattr(attn.q_norm, "variance_epsilon", eps))
            k = hf_norm(k, attn.k_norm.weight, getattr(attn.k_norm, "variance_epsilon", eps))
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            lt = cfg.layer_types[li]
            cos, sin = model.model.rotary_emb(x, pos[None, :], lt)  # HF's own, per layer type
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            rep = nH // nKV
            kk = k.repeat_interleave(rep, dim=1)   # [1, nH, T, D]
            vv = v.repeat_interleave(rep, dim=1)
            ki = torch.arange(T, device=dev).view(1, -1)
            CH = 512
            o_all = torch.empty(1, nH, T, D, dtype=q.dtype, device=dev)
            for c0 in range(0, T, CH):
                c1 = min(c0 + CH, T)
                qi = pos[c0:c1].view(-1, 1)
                maskc = ki <= qi
                if attn.sliding_window is not None and args.window_rule != "none":
                    if args.window_rule == "lt":
                        maskc = maskc & ((qi - ki) < W)
                    else:
                        maskc = maskc & ((qi - ki) <= W)
                sc = torch.matmul(q[:, :, c0:c1], kk.transpose(2, 3)) * scaling
                sc = sc.masked_fill(~maskc.view(1, 1, c1 - c0, T), float("-inf"))
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, :, c0:c1] = torch.matmul(pr, vv)
                del sc, pr
            o = o_all.transpose(1, 2).contiguous().view(1, T, nH * D)
            attn_out = F.linear(o, attn.o_proj.weight)
            hf_attn = hooks[(li, "attn")][0]
            d_attn = (attn_out[0].float() - hf_attn.float()).abs().max().item()
            x = x + attn_out
            del o, kk, vv, o_all

            h2 = hf_norm(x, layer.post_attention_layernorm.weight, eps)
            routed = h2.view(-1, cfg.hidden_size)
            logits_r = F.linear(routed, layer.mlp.gate.weight)
            probs_r = F.softmax(logits_r, dtype=torch.float, dim=-1)
            topv, topi = torch.topk(probs_r, cfg.num_experts_per_tok, dim=-1)
            if cfg.norm_topk_prob:
                topv = topv / topv.sum(dim=-1, keepdim=True)
            topv = topv.to(logits_r.dtype)
            moe_out = grouped_mm_experts_forward(
                layer.mlp.experts, routed, topi, topv
            ).view(1, T, cfg.hidden_size)
            d_mlp = (moe_out[0].float() - hooks[(li, "mlp")][0].float()).abs().max().item()
            x = x + moe_out
            d_block = (x[0].float() - hooks[(li, "block")][0].float()).abs().max().item()
            if d_block > 0 and first_bad is None:
                first_bad = li
            if li < 4 or (first_bad is not None and li <= first_bad + 1):
                print(f"L{li}: d_attn={d_attn:.6f} d_mlp={d_mlp:.6f} d_block={d_block:.6f} w={attn.sliding_window}", flush=True)
            del routed, h, h2, moe_out
            torch.cuda.empty_cache()

        x = hf_norm(x, model.model.norm.weight, eps)
        mine = F.linear(x.view(-1, cfg.hidden_size), model.lm_head.weight).view(T, -1)

    sel = torch.arange(P - 1, P + len(ref_tokens) - 1, device=dev)
    d = (mine[sel].float() - hf_logits[sel]).abs()
    print(f"max|delta| logits: {d.max().item():.6f}  mean: {d.mean().item():.8f}")
    print(f"argmax mismatches: {int((mine[sel].argmax(-1) != hf_logits[sel].argmax(-1)).sum())}/{len(sel)}")
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
