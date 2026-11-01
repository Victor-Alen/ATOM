#!/usr/bin/env python3
"""Replay layer-0 attention offline from the dumped q/k/v and compare with the engine's
own attention output, to separate mask/softmax from o_proj."""
import torch

d = torch.load("/workspace/logs/dumpl/l0_qk.pt")
q_last = d["q_last"].float()            # [nH, D]
k_all = d["k_all"].float()              # [ctx, nKV, D]
v_all = d["v_all"].float()              # [ctx, nKV, D]
nH, nKV, D = int(d["n_heads"]), int(d["n_kv"]), int(d["head_dim"])
W = int(d["window"])
scale = float(d["scale"])
o_last = torch.load("/workspace/logs/dumpl/l0_o_last.pt").float()  # [nH * D]

ctx, _, _ = k_all.shape
print(f"ctx={ctx} nH={nH} nKV={nKV} D={D} window={W} scale={scale:.6f}", flush=True)

rep = nH // nKV
pos = ctx - 1
kv_idx = torch.arange(ctx)
allowed = kv_idx <= pos
if W > 0:
    allowed = allowed & ((pos - kv_idx) < W)
allowed_upto = kv_idx <= pos
print(f"keys allowed by window={int(allowed.sum())} by causality only={int(allowed_upto.sum())}", flush=True)

out = torch.zeros(nH, D)
for h in range(nH):
    kvh = h // rep
    qh = q_last[h]
    kh = k_all[:, kvh, :]
    vh = v_all[:, kvh, :]
    sc = (qh @ kh.T) * scale
    sc = sc.masked_fill(~allowed, float("-inf"))
    p = torch.softmax(sc, dim=-1)
    out[h] = p @ vh

replay = out.reshape(-1)
print(f"replay vs engine o_last max|delta|={(replay - o_last).abs().max().item():.6f}", flush=True)
print(f"o absmax={o_last.abs().max().item():.4f} replay absmax={replay.abs().max().item():.4f}", flush=True)
print("### ATTENTION_REPLAY_DONE")
