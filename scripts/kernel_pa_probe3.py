#!/usr/bin/env python3
"""Probe v3: correct paged-layout scatter (verified against aiter's own
create_kv_cache / value shuffle), then
  E1 addressing check (K=0, V ramp -> uniform softmax must give mean token index)
  E2 determinism (identical inputs, repeated launches)
  E3 tail-leak: poison KV columns beyond ctx inside the last partial block
  E4 reference agreement in fp32
"""
import hashlib, json
import torch
import aiter, aiter.ops.attention
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon, get_recommended_splits

DEV, DT = "cuda", torch.bfloat16
BS, NQ, NKV, D = 16, 32, 4, 128
X = 16 // 2
HS = D // X
SCALE = 1.0 / (D ** 0.5)

def h(t):
    return hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:10]

def mk(k_log, v_log, nb):
    """k_log,v_log: [nb*BS, NKV, D] logical -> kernel paged layout."""
    kc = k_log.reshape(nb, BS, NKV, HS, X).permute(0, 2, 3, 1, 4).contiguous()
    vc = v_log.reshape(nb, BS // X, X, NKV, D).permute(0, 3, 1, 4, 2).contiguous()
    return kc, vc

def call(q, kc, vc, bt, ctx, splits, cps, window):
    lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    out = torch.empty(1, NQ, D, dtype=DT, device=DEV)
    pa_decode_gluon(out, q, kc, vc, lens, bt, SCALE, 1, splits, cps, DT,
                    None, None, None, None, None, None, None, None,
                    sliding_window=window, ps=True)
    return out.clone()

def ref(q, k_log, v_log, ctx, window):
    start = max(0, ctx - window) if window > 0 else 0
    k = k_log[start:ctx].float().permute(1, 0, 2)     # [kv,T,D]
    v = v_log[start:ctx].float().permute(1, 0, 2)
    qh = q[0].float().view(NKV, NQ // NKV, D)
    s = torch.einsum("kgd,ktd->kgt", qh, k) * SCALE
    p = torch.softmax(s, dim=-1)
    return torch.einsum("kgt,ktd->kgd", p, v).reshape(1, NQ, D).to(DT)

torch.manual_seed(5)
q = torch.randn(1, NQ, D, dtype=DT, device=DEV)
CFGS = (("A_swa_1x128", 1, 128, 1024), ("C_full_1x256", 1, 256, 0),
        ("B_full_%dx256" % get_recommended_splits(1, NKV), get_recommended_splits(1, NKV), 256, 0))

print("### E1 addressing: K=0, V[t,d]=t+1000d -> out[d]-1000d must equal mean token index")
for ctx in (16, 17, 33, 64, 65, 100, 128, 129, 150):
    nb = (ctx + BS - 1) // BS + 4
    t = torch.arange(nb * BS, device=DEV, dtype=torch.float32)
    v_log = (t[:, None] + 1000.0 * torch.arange(D, device=DEV)[None, :]).to(DT)
    v_log = v_log.view(nb * BS, 1, D).expand(nb * BS, NKV, D).contiguous()
    k_log = torch.zeros(nb * BS, NKV, D, dtype=DT, device=DEV)
    kc, vc = mk(k_log, v_log, nb)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    cells = []
    for name, sp, cps, win in CFGS:
        o = call(q, kc, vc, bt, ctx, sp, cps, win).float()
        imp = o[0, 0] - torch.arange(D, device=DEV) * 1000.0
        cells.append(f"{name}: mean={imp.mean().item():8.3f} std={imp.std().item():.3f}")
    print(f"ctx={ctx:4d} want_mean={(ctx-1)/2:7.2f} | " + " | ".join(cells), flush=True)

print("\n### E2 determinism + E4 reference agreement")
for ctx in (16, 32, 64, 65, 80, 96, 128, 129, 192, 256, 257):
    nb = (ctx + BS - 1) // BS + 4
    torch.manual_seed(11)
    k_log = torch.randn(nb * BS, NKV, D, dtype=DT, device=DEV)
    v_log = torch.randn(nb * BS, NKV, D, dtype=DT, device=DEV)
    kc, vc = mk(k_log, v_log, nb)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    cells = []
    for name, sp, cps, win in CFGS:
        hs, bs = set(), None
        for _ in range(8):
            o = call(q, kc, vc, bt, ctx, sp, cps, win)
            hs.add(h(o)); bs = o
        d = float((bs.float() - ref(q, k_log, v_log, ctx, win).float()).abs().max())
        cells.append(f"{name}: distinct={len(hs)} refdiff={d:.3e}")
    print(f"ctx={ctx:4d} | " + " | ".join(cells), flush=True)

print("\n### E3 tail leak: rewrite only KV columns >= ctx in the last partial block")
for ctx in (17, 33, 48, 64, 65, 66, 80, 96, 100, 129):
    nb = (ctx + BS - 1) // BS + 4
    nblk = (ctx + BS - 1) // BS
    keep = ctx - (nblk - 1) * BS
    torch.manual_seed(21)
    k_log = torch.randn(nb * BS, NKV, D, dtype=DT, device=DEV)
    v_log = torch.randn(nb * BS, NKV, D, dtype=DT, device=DEV)
    kc, vc = mk(k_log, v_log, nb)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    cells = []
    for name, sp, cps, win in CFGS:
        base = call(q, kc, vc, bt, ctx, sp, cps, win)
        worst_k = worst_v = 0.0
        for trial in range(3):
            g = torch.Generator(device=DEV).manual_seed(500 + trial)
            lo = (nblk - 1) * BS + keep           # first token past ctx
            k2 = k_log.clone(); v2 = v_log.clone()
            k2[lo:] = torch.randn(nb * BS - lo, NKV, D, generator=g, device=DEV, dtype=DT)
            v2[lo:] = torch.randn(nb * BS - lo, NKV, D, generator=g, device=DEV, dtype=DT)
            kc2, vc2 = mk(k2, v2, nb)
            ok = call(q, kc2, vc2, bt, ctx, sp, cps, win)
            worst_k = max(worst_k, float((ok.float() - base.float()).abs().max()))
        # V-only poison
        for trial in range(3):
            g = torch.Generator(device=DEV).manual_seed(700 + trial)
            lo = (nblk - 1) * BS + keep
            v2 = v_log.clone()
            v2[lo:] = torch.randn(nb * BS - lo, NKV, D, generator=g, device=DEV, dtype=DT)
            _, vc2 = mk(k_log, v2, nb)
            ov = call(q, kc, vc2, bt, ctx, sp, cps, win)
            worst_v = max(worst_v, float((ov.float() - base.float()).abs().max()))
        cells.append(f"{name}: Kdelta={worst_k:.4f} Vdelta={worst_v:.4f}")
    print(f"ctx={ctx:4d} keep={keep:2d} | " + " | ".join(cells), flush=True)
