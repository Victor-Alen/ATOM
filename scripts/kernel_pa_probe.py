#!/usr/bin/env python3
"""Isolated probe of the compiled gluon paged-attention decode kernel.

Question: is the reduction over KV blocks inside the kernel (a) deterministic and
(b) correct, at the 64-token boundary the engine diverges on?

Calls the very same python entry point that aiter registers as
torch.ops.aiter.pa_decode_gluon and that ATOM's attention_mha.py invokes.
"""
import os, sys, hashlib, json
import torch

import aiter
import aiter.ops.attention  # registers torch.ops.aiter.pa_decode_gluon
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon, get_recommended_splits

torch.manual_seed(1234)
DEV = "cuda"
DT = torch.bfloat16
BS = 16            # ATOM kv_cache_block_size default
NQ, NKV, D = 32, 4, 128
X = 16 // 2        # 16B elements for bf16
HEAD_SPLITS = D // X

def h(t):
    return hashlib.md5(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]

def build(ctx, nb_spare=4, fill=None):
    nb = (ctx + BS - 1) // BS + nb_spare
    if fill is None:
        fill = torch.randn
    k_log = torch.randn(nb, BS, NKV, D, dtype=DT, device=DEV)   # logical
    v_log = torch.randn(nb, BS, NKV, D, dtype=DT, device=DEV)
    k_cache = torch.empty(nb, NKV, HEAD_SPLITS, BS, X, dtype=DT, device=DEV)
    v_cache = torch.empty(nb, NKV, BS // X, D, X, dtype=DT, device=DEV)
    # scatter logical -> paged layout exactly as the kernel reads it
    k_cache.view(nb, NKV, HEAD_SPLITS, BS, X).copy_(
        k_log.permute(0, 2, 3, 1).reshape(nb, NKV, HEAD_SPLITS, BS, X).contiguous()
    )
    v_cache.copy_(
        v_log.permute(0, 2, 1, 3).reshape(nb, NKV, BS // X, D, X).contiguous()
    )
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    return k_log, v_log, k_cache, v_cache, bt, nb

def ref(q, k_log, v_log, ctx, scale, window):
    qf = q.float()
    ks, vs = [], []
    start = max(0, ctx - window) if window > 0 else 0
    nblk = (ctx + BS - 1) // BS
    for b in range(nblk):
        lo = max(start, b * BS)
        hi = min(ctx, (b + 1) * BS)
        if hi <= lo:
            continue
        ks.append(k_log[b, lo - b * BS:hi - b * BS])
        vs.append(v_log[b, lo - b * BS:hi - b * BS])
    k = torch.cat(ks, 0).float() if ks else torch.zeros(0, NKV, D, device=DEV)
    v = torch.cat(vs, 0).float() if vs else torch.zeros(0, NKV, D, device=DEV)
    qh = qf.view(NKV, NQ // NKV, D)          # [kv, group, D]
    kt = k.permute(1, 0, 2)                  # [kv, T, D]
    s = torch.einsum("kgd,ktd->kgt", qh, kt) * scale
    p = torch.softmax(s, dim=-1)
    vt = v.permute(1, 0, 2)                  # [kv, T, D]
    o = torch.einsum("kgt,ktd->kgd", p, vt)
    return o.reshape(1, NQ, D).to(DT)

def run_case(ctx, splits, cps, window, seed=0, repeats=8, poison=False):
    torch.manual_seed(seed)
    q = torch.randn(1, NQ, D, dtype=DT, device=DEV)
    k_log, v_log, k_cache, v_cache, bt, nb = build(ctx)
    lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    out = torch.empty(1, NQ, D, dtype=DT, device=DEV)
    scale = 1.0 / (D ** 0.5)
    def call():
        pa_decode_gluon(out, q, k_cache, v_cache, lens, bt, scale, 1,
                        splits, cps, DT, None, None, None,
                        None, None, None, None, None,
                        sliding_window=window, ps=True)
        return out.clone()
    hashes, outs = [], []
    for _ in range(repeats):
        o = call()
        hashes.append(h(o)); outs.append(o)
    base = outs[0]
    max_run_var = max(float((o.float() - base.float()).abs().max()) for o in outs[1:]) if repeats > 1 else 0.0
    r = ref(q, k_log, v_log, ctx, scale, window)
    diff = float((base.float() - r.float()).abs().max())
    res = dict(ctx=ctx, splits=splits, cps=cps, window=window,
               distinct=len(set(hashes)), run_var=max_run_var, ref_diff=diff)
    if poison:
        # rewrite ONLY the region beyond ctx with different garbage, keep valid tokens identical
        diffs = []
        for s2 in (1, 2, 3):
            torch.manual_seed(9000 + s2)
            k2, v2, kc2, vc2, bt2, _ = build(ctx)
            nblk = (ctx + BS - 1) // BS
            last = (ctx - 1) // BS
            keep = ctx - last * BS
            kc2.view(nb, NKV, HEAD_SPLITS, BS, X)[:, :, :, :keep, :] = \
                k_cache.view(nb, NKV, HEAD_SPLITS, BS, X)[:, :, :, :keep, :]
            vc2[:, :, :, :, :] = v_cache
            # restore valid columns for the last block of V too
            v_cache.view(nb, NKV, BS // X, D, X)
            o2 = torch.empty_like(out)
            pa_decode_gluon(o2, q, kc2, vc2, lens, bt2, scale, 1, splits, cps, DT,
                            None, None, None, None, None, None, None, None,
                            sliding_window=window, ps=True)
            diffs.append(float((o2.float() - base.float()).abs().max()))
        res["poison_max_delta"] = max(diffs)
    return res

spl_full = get_recommended_splits(1, NKV)
print(f"# gfx={torch.cuda.get_device_properties(0).gcnArchName} "
      f"recommended_splits(1,{NKV})={spl_full}", flush=True)
print("# configs: (A) SWA layer window=1024 splits=1 cps=128 | (B) full layer window=0 splits=%d cps=256 | (C) single-split full window=0 splits=1 cps=256" % spl_full, flush=True)
rows = []
for ctx in [16, 32, 48, 63, 64, 65, 80, 96, 127, 128, 129, 192, 255, 256, 257, 300]:
    for name, splits, cps, window in [("A", 1, 128, 1024), ("C", 1, 256, 0), ("B", spl_full, 256, 0)]:
        try:
            r = run_case(ctx, splits, cps, window, repeats=8,
                         poison=(name == "A" and ctx in (64, 65, 96)))
            r["cfg"] = name
            rows.append(r)
            print(f"cfg={name} ctx={ctx:4d} splits={splits} cps={cps} distinct={r['distinct']} "
                  f"run_var={r['run_var']:.3e} ref_diff={r['ref_diff']:.3e}"
                  + (f" poison_delta={r['poison_max_delta']:.3e}" if "poison_max_delta" in r else ""), flush=True)
        except Exception as e:
            print(f"cfg={name} ctx={ctx} EXC {type(e).__name__}: {e}", flush=True)
print("### JSON ###")
print(json.dumps(rows))
