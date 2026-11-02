#!/usr/bin/env python3
"""Three-way expert GEMM comparison for layer 0 on A-0006."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
d = torch.load("/workspace/logs/hf_moe_l0.pt")
h2 = d["h2"].to("cuda:0")
sel = d["sel"].to("cuda:0")
rw = d["rw"].to("cuda:0").to(torch.bfloat16)
moe_hf = d["moe"].to("cuda:0")

idx = json.load(open(Path(MODEL) / "model.safetensors.index.json"))
wm = idx["weight_map"]
names = [n for n in wm if n.startswith("model.layers.0.mlp.experts")]
print("checkpoint expert tensors:", [(n, wm[n]) for n in names], flush=True)

w = {}
for n in names:
    with safe_open(Path(MODEL) / wm[n], framework="pt") as fh:
        w[n.split(".")[-1]] = fh.get_tensor(n).to("cuda:0")
print("loaded:", {k: tuple(v.shape) for k, v in w.items()}, flush=True)

gu, dn = w["gate_up_proj"], w["down_proj"]
E, top_k = gu.shape[0], sel.shape[1]
tokens = h2.shape[0]


def run_grouped():
    token_idx = torch.arange(tokens, device="cuda:0").unsqueeze(1).expand(-1, top_k).reshape(-1)
    ids = sel.reshape(-1)
    x = h2[token_idx]
    perm = torch.argsort(ids)
    inv = torch.argsort(perm)
    xg, idsg, wg = x[perm], ids[perm], rw.reshape(-1)[perm]
    hist = torch.histc(idsg.int(), bins=E, min=0, max=E - 1)
    offs = torch.cumsum(hist, dim=0, dtype=torch.int32)
    gu_out = F.grouped_mm(xg, gu.transpose(-2, -1), offs=offs)
    g, u = gu_out.chunk(2, dim=-1)
    out = F.grouped_mm(F.silu(g) * u, dn.transpose(-2, -1), offs=offs)
    out = out * wg.unsqueeze(-1)
    return out[inv].view(tokens, top_k, -1).sum(dim=1)


def run_per_expert():
    buckets = {e: [] for e in range(E)}
    for t in range(tokens):
        for j in range(top_k):
            buckets[int(sel[t, j])].append((t, j))
    acc = torch.zeros(tokens, gu.shape[1] // 2, device="cuda:0", dtype=torch.bfloat16)
    for e, items in buckets.items():
        if not items:
            continue
        ts = torch.tensor([i[0] for i in items], device="cuda:0")
        js = torch.tensor([i[1] for i in items], device="cuda:0")
        seg = h2[ts]
        gu_o = F.linear(seg, gu[e])
        g, u = gu_o.chunk(2, dim=-1)
        o = F.linear(F.silu(g) * u, dn[e])
        acc.index_add_(0, ts, (o * rw[ts, js].unsqueeze(-1)).float())
    return acc


with torch.no_grad():
    moe_grouped = run_grouped().float()
    moe_per = run_per_expert().float()
    print(f"(a) HF moe       absmax={moe_hf.abs().max().item():.4f}", flush=True)
    print(f"(b) grouped_mm   vs HF  max|delta|={(moe_grouped - moe_hf).abs().max().item():.6g}", flush=True)
    print(f"(c) per-expert   vs HF  max|delta|={(moe_per - moe_hf).abs().max().item():.6g}", flush=True)
    print(f"(b) vs (c)              max|delta|={(moe_grouped - moe_per).abs().max().item():.6g}", flush=True)
print("### THREE_WAY_DONE", flush=True)
