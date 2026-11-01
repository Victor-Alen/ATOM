#!/usr/bin/env python3
"""Rebuild L0 q/k with the engine's own eps and compare again."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
d = torch.load("/workspace/logs/dumpl/l0_in.pt")
qk = torch.load("/workspace/logs/dumpl/l0_qk.pt")
h_ln = d["h_ln_last"].float()
eps = float(d["eps"])
q_last = qk["q_last"].float()
k_last = qk["k_last"].float()
cos = qk["cos_last"].float()
sin = qk["sin_last"].float()

idx = json.load(open(Path(MODEL) / "model.safetensors.index.json"))
wmap = idx["weight_map"]
want = {
    "model.layers.0.self_attn.q_proj.weight": "q",
    "model.layers.0.self_attn.k_proj.weight": "k",
    "model.layers.0.self_attn.q_norm.weight": "qn",
    "model.layers.0.self_attn.k_norm.weight": "kn",
}
t = {}
for name, key in want.items():
    with safe_open(Path(MODEL) / wmap[name], framework="pt") as fh:
        t[key] = fh.get_tensor(name)


def norm(x, w):
    f = x.float()
    v = f.pow(2).mean(-1, keepdim=True)
    return w * (f * torch.rsqrt(v + eps)).to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


h = h_ln.to(torch.bfloat16)
q = torch.nn.functional.linear(h, t["q"]).view(32, 128)
k = torch.nn.functional.linear(h, t["k"]).view(4, 128)
q = norm(q, t["qn"])
k = norm(k, t["kn"])
q = (q * cos.to(torch.bfloat16) + rotate_half(q) * sin.to(torch.bfloat16)).float()
k = (k * cos.to(torch.bfloat16) + rotate_half(k) * sin.to(torch.bfloat16)).float()
print(f"exact rebuild q max|delta|={(q - q_last).abs().max().item():.6f}")
print(f"exact rebuild k max|delta|={(k - k_last).abs().max().item():.6f}")
print(f"q absmax={q_last.abs().max().item():.4f}")
print("### REBUILD2_DONE")
