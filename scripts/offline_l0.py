#!/usr/bin/env python3
"""Offline consistency checks on the latched L0 dump (no GPU, no HF model)."""
import torch

d = torch.load("/workspace/logs/dumpl/l0_in.pt")
qk = torch.load("/workspace/logs/dumpl/l0_qk.pt")
emb = d["emb_last"].float()
h_ln = d["h_ln_last"].float()
w = d["ln_w"].float()
eps = float(d["eps"])
print("n_new =", int(d["n_new"]), "eps =", eps)
print("ids_last8 =", d["ids_last8"].tolist())

# engine norm formula: f32 normalise -> cast to bf16 -> multiply by (bf16) weight
f = emb
var = f.pow(2).mean(-1, keepdim=True)
normed = (f * torch.rsqrt(var + eps)).to(torch.bfloat16)
recon = (w.to(torch.bfloat16) * normed).float()
print(f"recon LN vs dumped h_ln max|delta|={(recon - h_ln).abs().max().item():.6f}")

hs = torch.load("/workspace/logs/hf_hs.pt")
hf_emb = hs[0].float()
print(f"emb_last vs HF embedding max|delta|={(emb - hf_emb).abs().max().item():.6f}")

f2 = hf_emb
var2 = f2.pow(2).mean(-1, keepdim=True)
n2 = (f2 * torch.rsqrt(var2 + eps)).to(torch.bfloat16)
recon_hf = (w.to(torch.bfloat16) * n2).float()
print(f"LN(HF emb) vs dumped h_ln max|delta|={(recon_hf - h_ln).abs().max().item():.6f}")
print(f"cos_last absmax={qk['cos_last'].abs().max().item():.4f} sin_last absmax={qk['sin_last'].abs().max().item():.4f}")
print("### OFFLINE_L0_DONE")
