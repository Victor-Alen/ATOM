#!/usr/bin/env python3
"""Verify the engine's layer-0 input equals embed(ids) and its post-norm value."""
import json

import torch
from safetensors import safe_open

blob = torch.load("/workspace/logs/atom_l0_in.pt", map_location="cpu")
ids = blob["ids"]
x_in = blob["x_in"]
h_ln = blob["h_ln"]
ln_w = blob["ln_w"]
print("ids:", ids[:8].tolist(), "n:", ids.numel(), "dtype:", ids.dtype)
print("x_in shape:", tuple(x_in.shape), "dtype:", x_in.dtype)
print("h_ln shape:", tuple(h_ln.shape), "dtype:", h_ln.dtype)

idx = json.loads(open("/data/models/Mellum2.1-12B-A2.5B-Thinking/model.safetensors.index.json").read())
wm = idx["weight_map"]


def get(n):
    with safe_open("/data/models/Mellum2.1-12B-A2.5B-Thinking/" + wm[n], framework="pt", device="cpu") as f:
        return f.get_tensor(n)


emb = get("model.embed_tokens.weight")
ref_x = emb[ids.long()]
print("x_in == embed(ids):", torch.equal(x_in, ref_x), "| max|d| =", (x_in.float() - ref_x.float()).abs().max().item())

f = ref_x.float()
var = f.pow(2).mean(-1, keepdim=True)
ref_h = ln_w * (f * torch.rsqrt(var + 1e-6)).to(ref_x.dtype)
print("h_ln == hf_norm(x_in):", torch.equal(h_ln, ref_h), "| max|d| =", (h_ln.float() - ref_h.float()).abs().max().item())
print("### DONE", flush=True)
