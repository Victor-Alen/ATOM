#!/usr/bin/env python3
"""Probe v2: decode the gluon PA kernel's actual K/V addressing and the leak at ctx=65."""
import hashlib, json
import torch
import aiter, aiter.ops.attention
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon, get_recommended_splits

DEV, DT = "cuda", torch.bfloat16
BS, NQ, NKV, D = 16, 32, 4, 128
X = 16 // 2
HS = D // X

def mk(k_log, v_log, nb):
    kc = torch.empty(nb, NKV, HS, BS, X, dtype=DT, device=DEV)
    vc = torch.empty(nb, NKV, BS // X, D, X, dtype=DT, device=DEV)
    kc.copy_(k_log.permute(0, 2, 3, 1).reshape(nb, NKV, HS, BS, X).contiguous())
    vc.copy_(v_log.permute(0, 2, 1, 3).reshape(nb, NKV, BS // X, D, X).contiguous())
    return kc, vc

def call(q, kc, vc, bt, ctx, splits, cps, window):
    lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    out = torch.empty(1, NQ, D, dtype=DT, device=DEV)
    pa_decode_gluon(out, q, kc, vc, lens, bt, 1.0 / (D ** 0.5), 1, splits, cps, DT,
                    None, None, None, None, None, None, None, None,
                    sliding_window=window, ps=True)
    return out.clone()

# ---------- experiment 1: V = ramp, K = 0  -> out = mean of attended token positions ----------
print("### E1: K=0, V[t,d]=t+1000*d ; out[d] - 1000*d = mean attended token index")
q = torch.randn(1, NQ, D, dtype=DT, device=DEV)
for ctx in (16, 32, 63, 64, 65, 80, 96, 127, 128, 129, 150):
    nb = (ctx + BS - 1) // BS + 6
    t = torch.arange(nb * BS, device=DEV, dtype=torch.float32)
    v_log = (t[:, None] + 1000.0 * torch.arange(D, device=DEV)[None, :]).to(DT)
    v_log = v_log.view(nb, BS, 1, D).expand(nb, BS, NKV, D).contiguous()
    k_log = torch.zeros(nb, BS, NKV, D, dtype=DT, device=DEV)
    kc, vc = mk(k_log, v_log, nb)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    row = []
    for name, splits, cps, win in (("A", 1, 128, 1024), ("C", 1, 256, 0), ("B", get_recommended_splits(1, NKV), 256, 0)):
        o = call(q, kc, vc, bt, ctx, splits, cps, win).float()
        d = torch.arange(D, device=DEV) * 1000.0
        implied = (o[0, 0] - d)           # one head sample
        row.append(f"{name}: implied_mean={implied.mean().item():7.3f} (want {(ctx-1)/2:7.3f}) "
                   f"spread={implied.std().item():.3f}")
    print(f"ctx={ctx:4d} | " + " | ".join(row), flush=True)

# ---------- experiment 2: locate the leak at ctx=65 ----------
print("\n### E2: cfg A (splits=1, cps=128, window=1024). Which region beyond ctx can change the output?")
def poison_test(ctx, mode):
    torch.manual_seed(7)
    nb = (ctx + BS - 1) // BS + 6
    nblk = (ctx + BS - 1) // BS
    k_log = torch.randn(nb, BS, NKV, D, dtype=DT, device=DEV)
    v_log = torch.randn(nb, BS, NKV, D, dtype=DT, device=DEV)
    kc, vc = mk(k_log, v_log, nb)
    bt = torch.arange(nb, dtype=torch.int32, device=DEV).unsqueeze(0)
    base = call(q, kc, vc, bt, ctx, 1, 128, 1024)
    rng = torch.Generator(device=DEV).manual_seed(99)
    out = []
    for trial in range(3):
        k2 = k_log.clone()
        keep = ctx - (nblk - 1) * BS          # valid columns in the last used block
        if mode in ("tail", "both"):
            k2[nblk - 1, keep:] = (torch.randn(nblk and (BS - keep), NKV, D, generator=rng, device=DEV, dtype=DT))
        if mode in ("spare", "both"):
            k2[nblk:] = torch.randn(nb - nblk, BS, NKV, D, generator=rng, device=DEV, dtype=DT)
        kc2, vc2 = mk(k2, v_log, nb)
        o = call(q, kc2, vc2, bt, ctx, 1, 128, 1024)
        out.append(float((o.float() - base.float()).abs().max()))
    return max(out)

for ctx in (33, 48, 64, 65, 66, 80, 96, 100):
    print(f"ctx={ctx:4d} tail-only delta={poison_test(ctx,'tail'):.4f}  "
          f"spare-only delta={poison_test(ctx,'spare'):.4f}  "
          f"both delta={poison_test(ctx,'both'):.4f}", flush=True)
