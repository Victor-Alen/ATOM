#!/usr/bin/env python3
"""Check the dumped V against v_proj applied to the dumped h_ln."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
d = torch.load("/workspace/logs/dumpl/l0_qk.pt")
info = torch.load("/workspace/logs/dumpl/l0_in.pt")
h_ln = info["h_ln_last"].float()
v_all = d["v_all"].float()
nKV = int(d["n_kv"])
D = int(d["head_dim"])

idx = json.load(open(Path(MODEL) / "model.safetensors.index.json"))
name = "model.layers.0.self_attn.v_proj.weight"
with safe_open(Path(MODEL) / idx["weight_map"][name], framework="pt") as fh:
    vw = fh.get_tensor(name)

exp = torch.nn.functional.linear(h_ln.to(torch.bfloat16), vw).view(nKV, D).float()
print(f"v_proj(h_ln) vs dumped v_all[-1] max|delta|={(exp - v_all[-1]).abs().max().item():.6f}")
print(f"v absmax={v_all[-1].abs().max().item():.4f} exp_absmax={exp.abs().max().item():.4f}")

# also check the q/k slices against the fused weight row order
qname = "model.layers.0.self_attn.q_proj.weight"
print("checking fused slice hypothesis is not needed if q/k matched earlier")
print("### V_CHECK_DONE")
