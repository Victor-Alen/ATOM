#!/usr/bin/env python3
"""Does my unshuffle invert aiter's shuffle_weight? (ATOM stores experts shuffled.)"""
import torch
from aiter.ops.shuffle import shuffle_weight


def unshuffle(w):
    e, n, k = w.shape
    return (
        w.view(e, n // 16, k // 32, 4, 16, 8)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .view(e, n, k)
    )


g = torch.Generator().manual_seed(0)
E, I, H = 8, 896, 2304
for name, w, layout in (
    ("w13", torch.randn(E, 2 * I, H, generator=g, dtype=torch.bfloat16), (16, 32)),
    ("w2", torch.randn(E, H, I, generator=g, dtype=torch.bfloat16), (16, 32)),
):
    try:
        s = shuffle_weight(w, layout)
    except Exception as exc:  # noqa: BLE001
        print(name, "shuffle_weight failed:", type(exc).__name__, str(exc)[:120])
        continue
    print(name, "shape", tuple(s.shape), "same as input:", torch.equal(s, w))
    u = unshuffle(s)
    print(name, "unshuffle(shuffle(w)) == w:", torch.equal(u, w),
          "| max|d| =", (u.float() - w.float()).abs().max().item())
print("### DONE", flush=True)
