#!/usr/bin/env python3
"""Compare engine embedding row with HF hidden_states[0] exactly."""
import torch

hs = torch.load("/workspace/logs/hf_hs.pt")
emb = torch.load("/workspace/logs/dumpl/emb.pt").float()
hf = hs[0].float()
d = (emb - hf).abs()
print(f"embedding max|delta|={d.max().item():.6f} equal={bool(torch.equal(emb, hf))}")
print(f"engine row absmax={emb.abs().max().item():.4f} hf row absmax={hf.abs().max().item():.4f}")
print("### EMB_CMP_DONE")
