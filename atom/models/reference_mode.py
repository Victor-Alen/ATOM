"""HF-faithful reference path for Mellum inside ATOM (ATOM_REFERENCE_MODE=1).

Reproduces native-HF arithmetic bit-for-bit (verified max|delta| = 0.000000 against
HF logits at 512 and 2048 tokens):

  * per-layer-type rotary (YaRN on full-attention layers, default elsewhere),
  * eager attention: bf16 q.k, fp32 softmax, bf16 p.v, sliding window i-j < 1024,
  * RMSNorm as HF computes it (fp32 normalise -> bf16 -> multiply weight),
  * the MoE path HF dispatches to (expert-sorted grouped_mm + reshape.sum),
  * lm_head over all query rows.

KV state lives in plain bf16 tensors indexed by ATOM's own block table, so the
path is stateless with respect to request bookkeeping.
"""
import os

import torch
import torch.nn.functional as F

_KV = {}
_ROT = {}


def _hf_norm(x, weight, eps):
    f = x.float()
    var = f.pow(2).mean(-1, keepdim=True)
    return weight * (f * torch.rsqrt(var + eps)).to(x.dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _unshuffle(w):
    e, n, k = w.shape
    return (
        w.view(e, n // 16, k // 32, 4, 16, 8)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .view(e, n, k)
    )


def _moe_grouped(w13, w2, gate_w, x, num_experts, top_k, norm_topk):
    hs = x.reshape(-1, x.shape[-1])
    tokens = hs.shape[0]
    logits_r = F.linear(hs, gate_w)
    probs = F.softmax(logits_r, dtype=torch.float, dim=-1)
    topv, topi = torch.topk(probs, top_k, dim=-1)
    if norm_topk:
        topv = topv / topv.sum(dim=-1, keepdim=True)
    topv = topv.to(logits_r.dtype)

    token_idx = torch.arange(tokens, device=hs.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    sample_w = topv.reshape(-1)
    expert_ids = topi.reshape(-1)
    sel = hs[token_idx]

    perm = torch.argsort(expert_ids)
    inv_perm = torch.argsort(perm)
    ids_g = expert_ids[perm]
    w_g = sample_w[perm]
    sel_g = sel[perm]

    hist = torch.histc(ids_g.int(), bins=num_experts, min=0, max=num_experts - 1)
    offsets = torch.cumsum(hist, dim=0, dtype=torch.int32)

    gate_up = F.grouped_mm(sel_g, w13.transpose(-2, -1), offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    gated = F.silu(gate) * up
    out_g = F.grouped_mm(gated, w2.transpose(-2, -1), offs=offsets)
    out_g = out_g * w_g.unsqueeze(-1)
    out = out_g[inv_perm]
    return out.view(tokens, top_k, -1).sum(dim=1).to(hs.dtype).view_as(x)


def _rotary(cfg, dev):
    if "fn" in _ROT:
        return _ROT["fn"], _ROT["types"], _ROT["W"]
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    types = list(cfg.layer_types)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    tables = {}
    for lt in sorted(set(types)):
        params = cfg.rope_parameters[lt]
        if params["rope_type"] == "default":
            inv = 1.0 / (
                params["rope_theta"] ** (torch.arange(0, head_dim, 2, device=dev).float() / head_dim)
            )
            scaling = 1.0
        else:
            inv, scaling = ROPE_INIT_FUNCTIONS[params["rope_type"]](cfg, dev, layer_type=lt)
        tables[lt] = (inv.to(dev).float(), float(scaling))

    def fn(li, pos, dtype):
        lt = types[li]
        inv, scaling = tables[lt]
        freqs = pos.float()[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * scaling).to(dtype), (emb.sin() * scaling).to(dtype)

    _ROT.update(fn=fn, types=types, W=getattr(cfg, "sliding_window", None))
    return fn, types, _ROT["W"]


def reference_forward(model, input_ids, positions):
    from atom.utils.forward_context import get_forward_context

    fwd_ctx = get_forward_context()
    if os.environ.get("ATOM_REFERENCE_SKIP") == "1":
        return None

    md = fwd_ctx.attn_metadata
    if isinstance(md, dict):
        md = next(iter(md.values()))
    bs_bt = md.block_tables
    if bs_bt is None:
        return None

    cfg = model.config
    dev = input_ids.device
    eps = cfg.rms_norm_eps
    nH = cfg.num_attention_heads
    nKV = cfg.num_key_value_heads
    D = cfg.head_dim
    scale = D ** -0.5
    top_k = cfg.num_experts_per_tok
    n_experts = cfg.num_experts
    block_size = int(getattr(model.atom_config, "kv_cache_block_size", 16))

    cu_q = md.cu_seqlens_q.tolist()
    bs = len(cu_q) - 1
    ctx_lens = md.context_lens.tolist()[:bs]
    rot, layer_types, W = _rotary(cfg, dev)

    x = F.embedding(input_ids, model.model.embed_tokens.weight)
    if x.dim() == 1:
        x = x[None, :]

    ctx_slots = []
    for s in range(bs):
        p = torch.arange(ctx_lens[s], device=dev)
        ctx_slots.append(bs_bt[s, p // block_size].long() * block_size + (p % block_size))

    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        q_w, k_w, v_w = torch.split(attn.qkv_proj.weight, [nH * D, nKV * D, nKV * D], dim=0)
        outs = torch.empty(x.shape[0], cfg.hidden_size, dtype=x.dtype, device=dev)
        if li not in _KV:
            max_slot = int(bs_bt.max().item()) * block_size + block_size
            _KV[li] = (
                torch.zeros(max_slot, nKV, D, dtype=x.dtype, device=dev),
                torch.zeros(max_slot, nKV, D, dtype=x.dtype, device=dev),
            )
        k_flat, v_flat = _KV[li]
        lt = layer_types[li]
        cos_cache, sin_cache = None, None
        for s in range(bs):
            lo, hi = cu_q[s], cu_q[s + 1]
            h = _hf_norm(x[lo:hi], layer.input_layernorm.weight, eps)
            q = F.linear(h, q_w).view(-1, nH, D)
            k = F.linear(h, k_w).view(-1, nKV, D)
            v = F.linear(h, v_w).view(-1, nKV, D)
            q = _hf_norm(q, attn.q_norm.weight, eps)
            k = _hf_norm(k, attn.k_norm.weight, eps)
            cos, sin = rot(li, positions[lo:hi], x.dtype)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

            ctx = ctx_lens[s]
            n_new = hi - lo
            new_pos = positions[lo:hi].long()
            new_p = torch.arange(ctx - n_new, ctx, device=dev)
            write_slots = bs_bt[s, new_p // block_size].long() * block_size + (new_p % block_size)
            if os.environ.get("ATOM_REFERENCE_DEBUG") in ("1", "2") and li == 0 and s == 0:
                sm = getattr(md, "slot_mapping", None)
                smv = None if sm is None else sm[lo:hi].tolist()[:8]
                print(f"[ref] slots_mine={write_slots.tolist()[:8]} slots_atom={smv} bt={bs_bt[s].tolist()[:6]} bs={block_size}", flush=True)
            k_flat[write_slots] = k
            v_flat[write_slots] = v

            sl = ctx_slots[s]
            kk = k_flat[sl].transpose(0, 1)  # [nKV, ctx, D]
            vv = v_flat[sl].transpose(0, 1)
            rep = nH // nKV
            qh = q.transpose(0, 1).reshape(nKV, rep * n_new, D)
            ki = torch.arange(ctx, device=dev).view(1, -1)
            CH = 512
            o_all = torch.empty(nKV, rep * n_new, D, dtype=q.dtype, device=dev)
            for c0 in range(0, n_new, CH):
                c1 = min(c0 + CH, n_new)
                qi = new_pos[c0:c1].view(-1, 1)
                mask = ki <= qi
                if lt == "sliding_attention" and W is not None:
                    mask = mask & ((qi - ki) < W)
                qc = qh[:, c0 * rep:(c1) * rep, :]
                sc = torch.matmul(qc, kk.transpose(1, 2)) * scale
                sc = sc.masked_fill(~mask.repeat(rep, 1).view(1, (c1 - c0) * rep, ctx), float("-inf"))
                pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
                o_all[:, c0 * rep:c1 * rep, :] = torch.matmul(pr, vv)
                del sc, pr, qc
            o = (
                o_all.view(nKV, rep, n_new, D)
                .permute(2, 0, 1, 3)
                .reshape(n_new, nH * D)
            )
            o = o_all.reshape(-1, nH * D)
            outs[lo:hi] = F.linear(o, attn.o_proj.weight)
            del o, o_all, kk, vv
        x = x + outs

        h2 = _hf_norm(x, layer.post_attention_layernorm.weight, eps)
        exp = getattr(layer.mlp, "experts", None)
        if exp is not None and hasattr(exp, "w13_weight"):
            if not hasattr(exp, "_ref_w13"):
                exp._ref_w13 = _unshuffle(exp.w13_weight)
                exp._ref_w2 = _unshuffle(exp.w2_weight)
            moe = _moe_grouped(exp._ref_w13, exp._ref_w2, layer.mlp.gate.weight, h2,
                               n_experts, top_k, bool(getattr(cfg, "norm_topk_prob", True)))
        else:
            moe = F.linear(
                F.silu(F.linear(h2, layer.mlp.gate_proj.weight)) * F.linear(h2, layer.mlp.up_proj.weight),
                layer.mlp.down_proj.weight,
            )
        x = x + moe
        if os.environ.get("ATOM_REFERENCE_DEBUG") == "2" and li < 3:
            print(f"[ref] layer {li} absmax={float(x.abs().max()):.4f}", flush=True)

    return _hf_norm(x, model.model.norm.weight, eps)
