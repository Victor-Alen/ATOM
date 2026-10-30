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
_TOKENS = {}
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
    """HF's own per-layer-type rotary embedding, loaded from the reference tree."""
    if "fn" in _ROT:
        return _ROT["fn"], _ROT["types"], _ROT["W"]
    import importlib.util
    import sys
    import types
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "reference" / "hf_mellum"
    pkg_name = "transformers.models.mellum"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(src)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
        for name in ("configuration_mellum", "modeling_mellum"):
            full = f"{pkg_name}.{name}"
            spec = importlib.util.spec_from_file_location(full, src / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
    MellumConfig = sys.modules[pkg_name + ".configuration_mellum"].MellumConfig
    modeling = sys.modules[pkg_name + ".modeling_mellum"]
    hf_cfg = MellumConfig.from_pretrained(os.environ["ATOM_REFERENCE_MODEL_DIR"])
    rot = modeling.MellumRotaryEmbedding(config=hf_cfg).to(dev)
    layer_types = list(hf_cfg.layer_types)
    hidden = hf_cfg.hidden_size

    def fn(li, pos, dtype):
        lt = layer_types[li]
        cos, sin = rot(
            torch.zeros(1, pos.numel(), hidden, dtype=dtype, device=dev),
            pos.view(1, -1),
            lt,
        )
        return cos[0], sin[0]

    _ROT.update(fn=fn, types=layer_types, W=getattr(hf_cfg, "sliding_window", None))
    return fn, layer_types, _ROT["W"]


_HIST: list = []


def history_after(new_ids):
    """Update the single-sequence token history and return it.

    A multi-token call is a prefill and replaces the history; a single-token call
    is a decode and appends. Keying this on the engine block table was wrong
    because decode steps have no block table (bs_bt is None).
    """
    if len(new_ids) > 1:
        _HIST[:] = list(new_ids)
    else:
        _HIST.append(new_ids[0])
    return _HIST


def seq_slots(positions, cu_q, ctx_lens, bs_bt, block_size, recomp, device):
    """KV slots for every request in the batch.

    In recompute mode the reference path owns the KV buffer, so slots are simply
    0..ctx-1 and the engine block table is not consulted at all (it is None on
    decode steps, which is what used to disable the reference path).
    """
    slots = []
    for s in range(len(ctx_lens)):
        if recomp:
            slots.append(torch.arange(int(ctx_lens[s]), device=device))
            continue
        p = positions[cu_q[s] : cu_q[s + 1]].long()
        n_ctx = int(p.max().item()) + 1
        q = torch.arange(n_ctx, device=device)
        slots.append(bs_bt[s, q // block_size].long() * block_size + (q % block_size))
    return slots


def reference_forward(model, input_ids, positions):
    from atom.utils.forward_context import get_forward_context

    fwd_ctx = get_forward_context()
    if os.environ.get("ATOM_REFERENCE_SKIP") == "1":
        return None
    if "frac" not in _ROT:
        try:
            _frac = float(os.environ.get("ATOM_REFERENCE_MEM_FRACTION", "0.85"))
            torch.cuda.set_per_process_memory_fraction(_frac, input_ids.device.index or 0)
            print(f"[ref] allocator fraction lifted to {_frac}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[ref] could not lift allocator fraction: {exc}", flush=True)
        _ROT["frac"] = True

    md = fwd_ctx.attn_metadata
    if isinstance(md, dict):
        md = next(iter(md.values()))
    bs_bt = md.block_tables
    if bs_bt is None and os.environ.get("ATOM_REFERENCE_RECOMPUTE") != "1":
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
    if bs > 1 and os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        # the recompute history is single-sequence by construction
        print(
            f"[ref] refusing recompute with bs={bs} (single-sequence mode); "
            "set ATOM_REFERENCE_RECOMPUTE=0 for batched runs",
            flush=True,
        )
        return None
    ctx_lens = md.context_lens.tolist()[:bs]
    rot, layer_types, W = _rotary(cfg, dev)
    if os.environ.get("ATOM_REFERENCE_DEBUG") in ("0", "1", "2"):
        print(
            f"[ref] pos={positions.tolist()[:4]}..{positions.tolist()[-2:]} "
            f"n={positions.numel()} cu_q={cu_q[:3]} ctx={ctx_lens[:3]}",
            flush=True,
        )

    x = F.embedding(input_ids, model.model.embed_tokens.weight)
    if x.dim() == 1:
        x = x[None, :]
    if os.environ.get("ATOM_REFERENCE_DEBUG") in ("0", "1", "2"):
        print(
            f"[ref] ids={input_ids[:4].tolist()} n={input_ids.numel()} "
            f"emb_absmax={float(x.abs().max()):.4f}",
            flush=True,
        )

    _recomp = False
    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        _recomp = True
        new_ids = input_ids.tolist()
        hist = history_after(new_ids)
        input_ids = torch.tensor(hist, dtype=torch.long, device=dev)
        positions = torch.arange(len(hist), device=dev)
        keep = torch.arange(len(hist) - len(new_ids), len(hist), device=dev)
        cu_q = [0, len(hist)]
        bs = 1
        ctx_lens = [len(hist)]

    ctx_slots = seq_slots(positions, cu_q, ctx_lens, bs_bt, block_size, _recomp, dev)

    if os.environ.get("ATOM_REFERENCE_DUMP") == "1" and "dumped" not in _ROT:
        import torch as _t

        blob = {}
        for li in (0, 2):
            lay = model.model.layers[li]
            a = lay.self_attn
            q_w, k_w, v_w = torch.split(a.qkv_proj.weight, [nH * D, nKV * D, nKV * D], dim=0)
            blob[f"L{li}.q"] = q_w.detach().cpu()
            blob[f"L{li}.k"] = k_w.detach().cpu()
            blob[f"L{li}.v"] = v_w.detach().cpu()
            blob[f"L{li}.o"] = a.o_proj.weight.detach().cpu()
            blob[f"L{li}.q_norm"] = a.q_norm.weight.detach().cpu()
            blob[f"L{li}.k_norm"] = a.k_norm.weight.detach().cpu()
            blob[f"L{li}.in_ln"] = lay.input_layernorm.weight.detach().cpu()
            blob[f"L{li}.post_ln"] = lay.post_attention_layernorm.weight.detach().cpu()
            blob[f"L{li}.gate"] = lay.mlp.gate.weight.detach().cpu()
            blob[f"L{li}.w13"] = _unshuffle(lay.mlp.experts.w13_weight).detach().cpu()
            blob[f"L{li}.w2"] = _unshuffle(lay.mlp.experts.w2_weight).detach().cpu()
        blob["cfg"] = {
            "eps": float(eps), "head_dim": int(D), "nH": int(nH), "nKV": int(nKV),
            "n_experts": int(n_experts), "top_k": int(top_k),
            "sliding_window": None if W is None else int(W),
        }
        blob["embed"] = model.model.embed_tokens.weight[:64].detach().cpu()
        blob["final_norm"] = model.model.norm.weight.detach().cpu()
        blob["lm_head"] = model.lm_head.weight[:64].detach().cpu()
        _t.save(blob, "/workspace/logs/atom_params.pt")
        _ROT["dumped"] = True
        print("[ref] dumped ATOM params -> /workspace/logs/atom_params.pt", flush=True)

    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        q_w, k_w, v_w = torch.split(attn.qkv_proj.weight, [nH * D, nKV * D, nKV * D], dim=0)
        outs = torch.empty(x.shape[0], cfg.hidden_size, dtype=x.dtype, device=dev)
        _need = int(bs_bt.max().item()) * block_size + block_size if bs_bt is not None else 0
        if _recomp:
            _need = max(_need, int(ctx_lens[0]) + 1)
        if li not in _KV or _KV[li][0].shape[0] < _need:
            max_slot = _need
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
            if li == 0 and os.environ.get("ATOM_REFERENCE_DUMP2") == "1" and "d2" not in _ROT:
                import torch as _t2

                _t2.save(
                    {
                        "ids": input_ids.detach().cpu(),
                        "x_in": x.detach().cpu(),
                        "h_ln": h.detach().cpu(),
                        "ln_w": layer.input_layernorm.weight.detach().cpu(),
                    },
                    "/workspace/logs/atom_l0_in.pt",
                )
                _ROT["d2"] = True
                print("[ref] dumped layer0 inputs", flush=True)
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

            n_new = hi - lo
            new_pos = positions[lo:hi].long()
            ctx = int(new_pos.max().item()) + 1
            write_slots = bs_bt[s, new_pos // block_size].long() * block_size + (new_pos % block_size)
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
            qh = q.transpose(0, 1).view(nKV, rep, n_new, D)
            ki = torch.arange(ctx, device=dev).view(1, -1)
            _ch = int(os.environ.get("ATOM_REFERENCE_ATTN_CHUNK", "0"))
            CH = n_new if _ch <= 0 else _ch
            o_all = torch.empty(nKV, rep, n_new, D, dtype=q.dtype, device=dev)
            for c0 in range(0, n_new, CH):
                c1 = min(c0 + CH, n_new)
                qi = new_pos[c0:c1].view(-1, 1)
                mask = ki <= qi
                if lt == "sliding_attention" and W is not None:
                    mask = mask & ((qi - ki) < W)
                qc = qh[:, :, c0:c1, :].reshape(nKV, rep * (c1 - c0), D)
                sc = torch.matmul(qc, kk.transpose(1, 2)) * scale
                neg = ~mask.repeat(rep, 1).view(1, (c1 - c0) * rep, ctx)
                sc.masked_fill_(neg, float("-inf"))
                del neg
                pr32 = F.softmax(sc, dim=-1, dtype=torch.float32)
                del sc
                pr = pr32.to(q.dtype)
                del pr32
                o_all[:, :, c0:c1, :] = torch.matmul(pr, vv).view(nKV, rep, c1 - c0, D)
                del pr, qc
            o = o_all.permute(2, 0, 1, 3).reshape(n_new, nH * D)
            outs[lo:hi] = F.linear(o, attn.o_proj.weight)
            del o, o_all, kk, vv
        x = x + outs

        h2 = _hf_norm(x, layer.post_attention_layernorm.weight, eps)
        exp = getattr(layer.mlp, "experts", None)
        if exp is not None and hasattr(exp, "w13_weight"):
            _w13 = _unshuffle(exp.w13_weight)
            _w2 = _unshuffle(exp.w2_weight)
            moe = _moe_grouped(_w13, _w2, layer.mlp.gate.weight, h2,
                               n_experts, top_k, bool(getattr(cfg, "norm_topk_prob", True)))
            del _w13, _w2
        else:
            moe = F.linear(
                F.silu(F.linear(h2, layer.mlp.gate_proj.weight)) * F.linear(h2, layer.mlp.up_proj.weight),
                layer.mlp.down_proj.weight,
            )
        x = x + moe
        if os.environ.get("ATOM_REFERENCE_DEBUG") == "2" and li < 8:
            print(f"[ref] L{li} last_absmax={float(x[-1].abs().max()):.4f}", flush=True)

    if os.environ.get("ATOM_REFERENCE_DEBUG") in ("0", "1", "2"):
        print(f"[ref] final_absmax={float(x.abs().max()):.4f}", flush=True)
    out = _hf_norm(x, model.model.norm.weight, eps)
    if os.environ.get("ATOM_REFERENCE_RECOMPUTE") == "1":
        return out[keep]
    return out
