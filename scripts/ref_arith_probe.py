#!/usr/bin/env python3
"""Probe: can an independent torch forward reproduce HF Mellum bit-exactly?

Single-shot full-sequence forward (same op order and shapes as HF eager), then
compare the logits at the reference positions against HF's own teacher-forced
logits. If max|delta| == 0 the exact-arithmetic route is proven.
"""
import argparse
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
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--window-rule", default="lt", choices=["lt", "le", "none"])
    ap.add_argument("--prompt-tokens", type=int, default=512)
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

    with torch.no_grad():
        out = model(input_ids=ids, logits_to_keep=int(T))
    hf_logits = out.logits[0].float()
    del out
    torch.cuda.empty_cache()
    print("HF logits ready", flush=True)

    cfg = model.config
    dev = args.device
    eps = cfg.rms_norm_eps
    theta = 500000.0
    nH, nKV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    scaling = D ** -0.5
    W = cfg.sliding_window
    layers = model.model.layers

    pos = torch.arange(T, device=dev)
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=dev).float() / D))
    freqs = pos.float()[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16)[:, None, :]  # [T,1,D]
    sin = emb.sin().to(torch.bfloat16)[:, None, :]

    j = torch.arange(T, device=dev)
    causal = j[None, :] <= pos[:, None]

    with torch.no_grad():
        x = F.embedding(ids, model.model.embed_tokens.weight)  # [1,T,H]
        for li, layer in enumerate(layers):
            attn = layer.self_attn
            h = hf_norm(x, layer.input_layernorm.weight, eps)
            q = F.linear(h, attn.q_proj.weight).view(1, T, nH, D)
            k = F.linear(h, attn.k_proj.weight).view(1, T, nKV, D)
            v = F.linear(h, attn.v_proj.weight).view(1, T, nKV, D)
            q = hf_norm(q, attn.q_norm.weight, getattr(attn.q_norm, "variance_epsilon", eps))
            k = hf_norm(k, attn.k_norm.weight, getattr(attn.k_norm, "variance_epsilon", eps))
            q = q.transpose(1, 2)  # [1,nH,T,D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos.permute(1, 0, 2) + rotate_half(q) * sin.permute(1, 0, 2)
            k = k * cos.permute(1, 0, 2) + rotate_half(k) * sin.permute(1, 0, 2)
            kk = k.repeat_interleave(nH // nKV, dim=1)
            vv = v.repeat_interleave(nH // nKV, dim=1)
            scores = torch.matmul(q, kk.transpose(2, 3)) * scaling  # [1,nH,T,T] bf16
            mask = causal
            if attn.sliding_window is not None and args.window_rule != "none":
                if args.window_rule == "lt":
                    mask = mask & ((pos[:, None] - j[None, :]) < W)
                else:
                    mask = mask & ((pos[:, None] - j[None, :]) <= W)
            scores = scores.masked_fill(~mask.view(1, 1, T, T), float("-inf"))
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            o = torch.matmul(probs, vv)  # [1,nH,T,D]
            o = o.transpose(1, 2).contiguous().view(1, T, nH * D)
            x = x + F.linear(o, attn.o_proj.weight)
            del scores, probs, o, kk, vv

            h2 = hf_norm(x, layer.post_attention_layernorm.weight, eps)
            routed = h2.view(-1, cfg.hidden_size)
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
                inter = F.silu(g) * u
                outp = F.linear(inter, dn[e]) * topv[ti, si, None]
                final.index_add_(0, ti, outp.to(final.dtype))
            x = x + final.view(1, T, cfg.hidden_size)
            del routed, final, h, h2
            torch.cuda.empty_cache()
            if li % 7 == 0:
                print(f"layer {li} done", flush=True)

        x = hf_norm(x, model.model.norm.weight, eps)
        mine = F.linear(x.view(-1, cfg.hidden_size), model.lm_head.weight).view(T, -1)

    sel = torch.arange(P - 1, P + len(ref_tokens) - 1, device=dev)
    d = (mine[sel].float() - hf_logits[sel]).abs()
    print(f"positions compared: {len(sel)}")
    print(f"max|delta|: {d.max().item():.6f}")
    print(f"mean|delta|: {d.mean().item():.8f}")
    mism = int((mine[sel].argmax(-1) != hf_logits[sel].argmax(-1)).sum())
    print(f"argmax mismatches: {mism}/{len(sel)}")
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
