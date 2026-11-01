#!/usr/bin/env python3
"""Close the loop on layer 0: rebuild the attention residual offline and compare with
both the engine's own dump and the HF-captured value."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
info = torch.load("/workspace/logs/dumpl/l0_in.pt")
d = torch.load("/workspace/logs/dumpl/l0_qk.pt")
o = torch.load("/workspace/logs/dumpl/l0_o_last.pt").float()
emb = info["emb_last"].float()

idx = json.load(open(Path(MODEL) / "model.safetensors.index.json"))
name = "model.layers.0.self_attn.o_proj.weight"
with safe_open(Path(MODEL) / idx["weight_map"][name], framework="pt") as fh:
    ow = fh.get_tensor(name)

attn_out = torch.nn.functional.linear(o.to(torch.bfloat16), ow).float()
recon = emb + attn_out
try:
    eng_attn = torch.load("/workspace/logs/dumpl/attn_L0.pt").float()
    print(f"recon(emb + o_proj(o)) vs engine attn_L0 max|delta|={(recon - eng_attn).abs().max().item():.6f}")
except FileNotFoundError:
    print("engine attn_L0 dump missing")

try:
    hf_attn = torch.load("/workspace/logs/hf_attn_L0.pt").float()
    print(f"engine attn_L0 vs HF attn residual max|delta|={(eng_attn - hf_attn).abs().max().item():.6f}")
    print(f"recon vs HF attn residual max|delta|={(recon - hf_attn).abs().max().item():.6f}")
    print(f"engine absmax={eng_attn.abs().max().item():.4f} hf absmax={hf_attn.abs().max().item():.4f}")
except FileNotFoundError:
    print("hf_attn_L0.pt missing")
print("### CLOSE_L0_DONE")
