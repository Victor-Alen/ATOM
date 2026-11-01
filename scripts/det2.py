#!/usr/bin/env python3
"""Is the engine's reference forward deterministic run to run?"""
import torch

a = torch.load("/workspace/logs/dumpl/layers_0.pt")
b = torch.load("/workspace/logs/dumpl/layers_1.pt")
print(f"entries: {len(a)} vs {len(b)}")
first = None
for li in range(28):
    ka, kb = f"blk_{li}", f"blk_{li}"
    if ka not in a or kb not in b:
        continue
    d = (a[ka] - b[kb]).abs().max().item()
    if d != 0.0:
        print(f"L{li:02d} run-to-run diff {d:.6f}")
        if first is None:
            first = li
print(f"first layer differing between two identical runs: {first}")
fa, fb = a.get("final"), b.get("final")
if fa is not None and fb is not None:
    print(f"final run-to-run max|delta|={(fa - fb).abs().max().item():.6f}")
print("### DET2_DONE")
