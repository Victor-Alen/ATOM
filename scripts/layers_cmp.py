#!/usr/bin/env python3
"""Compare every layer of the latched dump with HF's hidden_states (same prompt)."""
import torch

d = torch.load("/workspace/logs/dumpl/layers.pt")
hs = torch.load("/workspace/logs/hf_hs.pt")  # 29 last-row tensors
print(f"dumped layers: {len(d)} entries, hf: {len(hs)}", flush=True)

first = None
for li in range(len(hs) - 1):
    key = f"blk_{li}"
    if key not in d:
        continue
    eng = d[key].float()
    hf = hs[li + 1].float()
    delta = (eng - hf).abs().max().item()
    status = "exact" if delta == 0.0 else f"DIFF {delta:.6f}"
    print(f"L{li:02d} {status}", flush=True)
    if delta != 0.0 and first is None:
        first = li

fin = d.get("final")
if fin is not None:
    hf_final = hs[-1].float()
    print(f"final norm max|delta|={(fin.float() - hf_final).abs().max().item():.6f}", flush=True)
print(f"first differing layer: {first}", flush=True)
print("### LAYERS_CMP_DONE")
