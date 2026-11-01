#!/usr/bin/env python3
"""Offline: compare engine L0 dumps with HF's cached per-layer hidden, and locate
which sub-op (embedding / attention / mlp) first differs."""
import torch

hs = torch.load("/workspace/logs/hf_hs.pt")  # list of 29 last-row tensors
eng_in = torch.load("/workspace/logs/dumpl/in_L0.pt").float()
eng_attn = torch.load("/workspace/logs/dumpl/attn_L0.pt").float()
eng_blk = torch.load("/workspace/logs/dumpl/blk_L0.pt").float()
hf_emb = hs[0].float()
hf_blk0 = hs[1].float()

print(f"embedding   max|delta|={(eng_in - hf_emb).abs().max().item():.6f}")
print(f"L0 block out max|delta|={(eng_blk - hf_blk0).abs().max().item():.6f}")

# HF's layer-0 attention output for comparison: block_out - mlp_out is not directly
# available, so also report the engine's own attn contribution statistics.
attn_contrib = (eng_attn - eng_in).abs().max().item()
print(f"engine|attn_contrib|max={attn_contrib:.6f}")
print("### OFFLINE_CMP_DONE")
