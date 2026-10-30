#!/usr/bin/env python3
"""End-to-end arithmetic-equivalence check on ATOM's packed layouts.

1. take HF's own weights, pack them exactly the way ATOM stores them
   (fused qkv, aiter-shuffled MoE),
2. run the reference forward using ONLY those packed tensors (splitting / unshuffling),
3. greedy-decode 64 tokens and compare with HF's reference tokens.
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


def shuffle(w):
    e, n, k = w.shape
    return (
        w.view(e, n // 16, 16, k // 32, 4, 8)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(e, n, k)
    )


def unshuffle(w):
    e, n, k = w.shape
    return (
        w.view(e, n // 16, k // 32, 4, 16, 8)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .view(e, n, k)
    )


def hf_norm(x, w, eps):
    f = x.float()
    v = f.pow(2).mean(-1, keepdim=True)
    return w * (f * torch.rsqrt(v + eps)).to(x.dtype)


def rot_half(x):
    a = x[..., : x.shape[-1] // 2]
    b = x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), dim=-1)


def moe_grouped(w13, w2, gate_w, x, E, k, norm_topk):
    hs = x.reshape(-1, x.shape[-1])
    n = hs.shape[0]
    lg = F.linear(hs, gate_w)
    pr = F.softmax(lg, dtype=torch.float, dim=-1)
    tv, ti = torch.topk(pr, k, dim=-1)
    if norm_topk:
        tv = tv / tv.sum(dim=-1, keepdim=True)
    tv = tv.to(lg.dtype)

    tok = torch.arange(n, device=hs.device).unsqueeze(1).expand(-1, k).reshape(-1)
    sw = tv.reshape(-1)
    eid = ti.reshape(-1)
    sel = hs[tok]
    perm = torch.argsort(eid)
    inv = torch.argsort(perm)
    eid_g = eid[perm]
    sw_g = sw[perm]
    sel_g = sel[perm]
    hist = torch.histc(eid_g.int(), bins=E, min=0, max=E - 1)
    offs = torch.cumsum(hist, dim=0, dtype=torch.int32)
    gu = F.grouped_mm(sel_g, w13.transpose(-2, -1), offs=offs)
    g, u = gu.chunk(2, dim=-1)
    out = F.grouped_mm(F.silu(g) * u, w2.transpose(-2, -1), offs=offs) * sw_g.unsqueeze(-1)
    return out[inv].view(n, k, -1).sum(dim=1).to(hs.dtype).view_as(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-id", default="C-0000")
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    ref = load_reference_module()
    MellumConfig, MellumForCausalLM = ref.load_mellum_classes()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    cfg_hf = MellumConfig.from_pretrained(MODEL_PATH)
    cfg_hf._attn_implementation = "eager"
    print("loading HF ...", flush=True)
    hf = MellumForCausalLM.from_pretrained(
        MODEL_PATH, config=cfg_hf, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    ref.load_checkpoint_experts(hf, MODEL_PATH)
    hf = hf.to(args.device).eval()
    print("HF ready", flush=True)

    rec = None
    for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["id"] == args.prompt_id:
            rec = r
            break
    prompt = rec["prompt_token_ids"][: args.prompt_tokens]
    hf_ref_tokens = rec["token_ids"][: args.steps]

    cfg = hf.config
    dev = args.device
    eps = cfg.rms_norm_eps
    nH, nKV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    scale = D ** -0.5
    W = cfg.sliding_window
    layer_types = list(cfg.layer_types)
    E, top_k = cfg.num_experts, cfg.num_experts_per_tok

    # ---- pack HF weights into ATOM's layout ----
    L = []
    with torch.no_grad():
        for layer in hf.model.layers:
            a = layer.self_attn
            qkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0).contiguous()
            L.append({
                "qkv": qkv,
                "o": a.o_proj.weight.contiguous(),
                "q_norm": a.q_norm.weight,
                "k_norm": a.k_norm.weight,
                "in_ln": layer.input_layernorm.weight,
                "post_ln": layer.post_attention_layernorm.weight,
                "gate": layer.mlp.gate.weight.contiguous(),
                "w13": shuffle(layer.mlp.experts.gate_up_proj.contiguous()),
                "w2": shuffle(layer.mlp.experts.down_proj.contiguous()),
            })
    embed = hf.model.embed_tokens.weight
    final_norm = hf.model.norm.weight
    lm_head = hf.lm_head.weight
    print("packed weights for", len(L), "layers", flush=True)

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    head_dim = D
    tables = {}
    for lt in sorted(set(layer_types)):
        p = cfg.rope_parameters[lt]
        if p["rope_type"] == "default":
            inv = 1.0 / (p["rope_theta"] ** (torch.arange(0, head_dim, 2, device=dev).float() / head_dim))
            sc = 1.0
        else:
            inv, sc = ROPE_INIT_FUNCTIONS[p["rope_type"]](cfg, dev, layer_type=lt)
        tables[lt] = (inv.to(dev).float(), float(sc))

    def rope(li, pos, dtype):
        inv, sc = tables[layer_types[li]]
        f = pos.float()[:, None] * inv[None, :]
        emb = torch.cat((f, f), dim=-1)
        return (emb.cos() * sc).to(dtype), (emb.sin() * sc).to(dtype)

    caches = [None] * len(L)
    out_tokens = []
    cur = list(prompt)
    total = len(prompt) + args.steps
    for t in range(len(prompt), total):
        ids = torch.tensor([cur], dtype=torch.long, device=dev)
        pos = torch.arange(len(cur), device=dev)
        qlo = len(cur) - 1 if out_tokens else 0
        with torch.no_grad():
            x = F.embedding(ids, embed)[0]
            for li, w in enumerate(L):
                q_w, k_w, v_w = torch.split(w["qkv"], [nH * D, nKV * D, nKV * D], dim=0)
                h = hf_norm(x[qlo:], w["in_ln"], eps)
                q = F.linear(h, q_w).view(-1, nH, D)
                k = F.linear(h, k_w).view(-1, nKV, D)
                v = F.linear(h, v_w).view(-1, nKV, D)
                q = hf_norm(q, w["q_norm"], eps)
                k = hf_norm(k, w["k_norm"], eps)
                cos, sin = rope(li, pos[qlo:], x.dtype)
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
                q = (q * cos + rot_half(q) * sin).transpose(0, 1)
                k = (k * cos + rot_half(k) * sin).transpose(0, 1)
                v = v.transpose(0, 1)
                ck = caches[li]
                if ck is None:
                    caches[li] = (k, v)
                else:
                    caches[li] = (torch.cat([ck[0], k], 1), torch.cat([ck[1], v], 1))
                kk, vv = caches[li]
                S = kk.shape[1]
                qh = q.reshape(nKV, (nH // nKV) * q.shape[1], D)
                ki = torch.arange(S, device=dev).view(1, -1)
                qi = pos[qlo:].view(-1, 1)
                mask = ki <= qi
                if layer_types[li] == "sliding_attention" and W is not None:
                    mask = mask & ((qi - ki) < W)
                sc_ = torch.matmul(qh, kk.transpose(1, 2)) * scale
                sc_ = sc_.masked_fill(~mask.repeat(nH // nKV, 1).view(1, -1, S), float("-inf"))
                pr = F.softmax(sc_, dim=-1, dtype=torch.float32).to(q.dtype)
                o = torch.matmul(pr, vv)
                o = o.view(nKV, nH // nKV, -1, D).permute(2, 0, 1, 3).reshape(-1, nH * D)
                x = x[qlo:] + F.linear(o, w["o"])
                h2 = hf_norm(x, w["post_ln"], eps)
                moe = moe_grouped(unshuffle(w["w13"]), unshuffle(w["w2"]), w["gate"], h2, E, top_k, True)
                x = x + moe
                del q, k, v, o, pr, sc_, moe, h2, h
            x = hf_norm(x, final_norm, eps)
            logits = F.linear(x[-1:], lm_head)
        nxt = int(logits.argmax(-1))
        out_tokens.append(nxt)
        cur.append(nxt)
        if (len(out_tokens) % 8) == 0:
            print(f"step {len(out_tokens)}: {out_tokens[-8:]}", flush=True)

    eq = sum(1 for a, b in zip(out_tokens, hf_ref_tokens) if a == b)
    print("atom-packed:", out_tokens[:12])
    print("hf         :", hf_ref_tokens[:12])
    print(f"identical: {out_tokens == hf_ref_tokens} | positions equal: {eq}/{len(hf_ref_tokens)}")
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
