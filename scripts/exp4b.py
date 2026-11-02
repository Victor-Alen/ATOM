#!/usr/bin/env python3
"""Three-way expert GEMM comparison (weights assembled from per-expert checkpoint tensors)."""
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
E = 64
gu_parts, dn_parts = [], []
files = {wm[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] for e in range(E)}
handles = {f: safe_open(Path(MODEL) / f, framework="pt") for f in files}
for e in range(E):
    g = handles[wm[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]].get_tensor(
        f"model.layers.0.mlp.experts.{e}.gate_proj.weight")
    u = handles[wm[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]].get_tensor(
        f"model.layers.0.mlp.experts.{e}.up_proj.weight")
    dn = handles[wm[f"model.layers.0.mlp.experts.{e}.down_proj.weight"]].get_tensor(
        f"model.layers.0.mlp.experts.{e}.down_proj.weight")
    gu_parts.append(torch.cat([g, u], dim=0))
    dn_parts.append(dn)
for h in handles.values():
    h.__exit__(None, None, None)
gu = torch.stack(gu_parts).to("cuda:0").to(torch.bfloat16)
dn = torch.stack(dn_parts).to("cuda:0").to(torch.bfloat16)
print("assembled gate_up", tuple(gu.shape), "down", tuple(dn.shape), flush=True)

tokens, top_k = h2.shape[0], sel.shape[1]


def run_grouped():
    token_idx = torch.arange(tokens, device="cuda:0").unsqueeze(1).expand(-1, top_k).reshape(-1)
    ids = sel.reshape(-1)
    x = h2[token_idx]
    perm = torch.argsort(ids)
    inv = torch.argsort(perm)
    xg, idsg, wg = x[perm], ids[perm], rw.reshape(-1)[perm]
    hist = torch.histc(idsg.int(), bins=E, min=0, max=E - 1)
    offs = torch.cumsum(hist, dim=0, dtype=torch.int32)
    o = F.grouped_mm(xg, gu.transpose(-2, -1), offs=offs)
    g, u = o.chunk(2, dim=-1)
    out = F.grouped_mm(F.silu(g) * u, dn.transpose(-2, -1), offs=offs)
    return (out * wg.unsqueeze(-1))[inv].view(tokens, top_k, -1).sum(dim=1).float()


def run_per_expert():
    acc = torch.zeros(tokens, dn.shape[-2], device="cuda:0", dtype=torch.float32)
    for e in range(E):
        mask = sel == e
        ts, js = mask.nonzero(as_tuple=True)
        if ts.numel() == 0:
            continue
        seg = h2[ts]
        o = F.linear(seg, gu[e])
        g, u = o.chunk(2, dim=-1)
        out = F.linear(F.silu(g) * u, dn[e])
        acc.index_add_(0, ts, out.float() * rw[ts, js].float().unsqueeze(-1))
    return acc


with torch.no_grad():
    a = moe_hf.float()
    b = run_grouped()
    c = run_per_expert()
    print(f"HF absmax={a.abs().max().item():.4f}", flush=True)
    print(f"grouped_mm vs HF   max|delta| = {(b - a).abs().max().item():.6g}", flush=True)
    print(f"per-expert vs HF   max|delta| = {(c - a).abs().max().item():.6g}", flush=True)
    print(f"grouped vs per-exp max|delta| = {(b - c).abs().max().item():.6g}", flush=True)
print("### THREE_WAY2_DONE", flush=True)
