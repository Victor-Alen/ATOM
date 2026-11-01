#!/usr/bin/env python3
"""Offline: rebuild L0 q/k from HF's checkpoint weights and the dumped h_ln/cos/sin."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
d = torch.load("/workspace/logs/dumpl/l0_in.pt")
qk = torch.load("/workspace/logs/dumpl/l0_qk.pt")
h_ln = d["h_ln_last"].float()
q_last = qk["q_last"].float()
k_last = qk["k_last"].float()
cos = qk["cos_last"].float()
sin = qk["sin_last"].float()

idx = json.load(open(Path(MODEL) / "model.safetensors.index.json"))
wmap = idx["weight_map"]
want = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.k_norm.weight",
]
tensors = {}
files = {wmap[n] for n in want}
for fname in files:
    with safe_open(Path(MODEL) / fname, framework="pt") as fh:
        for n in want:
            if wmap.get(n) == fname:
                tensors[n] = fh.get_tensor(n).float()
print("loaded:", sorted(tensors), flush=True)


def rms(x, w, eps=1e-6):
    f = x.float()
    v = f.pow(2).mean(-1, keepdim=True)
    return w * (f * torch.rsqrt(v + eps))


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


h = h_ln.to(torch.bfloat16)
q = torch.nn.functional.linear(h, tensors["model.layers.0.self_attn.q_proj.weight"].to(torch.bfloat16))
k = torch.nn.functional.linear(h, tensors["model.layers.0.self_attn.k_proj.weight"].to(torch.bfloat16))
q = q.view(32, 128)
k = k.view(4, 128)
q = rms(q, tensors["model.layers.0.self_attn.q_norm.weight"].to(torch.bfloat16)).float()
k = rms(k, tensors["model.layers.0.self_attn.k_norm.weight"].to(torch.bfloat16)).float()
q = (q * cos + rotate_half(q) * sin)
k = (k * cos + rotate_half(k) * sin)
print(f"rebuild q vs engine q_last max|delta|={(q - q_last).abs().max().item():.6f}", flush=True)
print(f"rebuild k vs engine k_last max|delta|={(k - k_last).abs().max().item():.6f}", flush=True)
print(f"q absmax={q.abs().max().item():.4f} k absmax={k.abs().max().item():.4f}", flush=True)
print("### REBUILD_DONE")
