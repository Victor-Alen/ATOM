#!/usr/bin/env python3
"""Verify unshuffle/ shuffle are exact inverses (index-derived, not hand-written)."""
import torch


def unshuffle(w):
    e, n, k = w.shape
    return (
        w.view(e, n // 16, k // 32, 4, 16, 8)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .view(e, n, k)
    )


def shuffle(w):
    e, n, k = w.shape
    return (
        w.view(e, n // 16, 16, k // 32, 4, 8)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(e, n, k)
    )


g = torch.Generator(device="cuda:1").manual_seed(0)
E, I, H = 64, 896, 2304
w13 = torch.randn(E, 2 * I, H, generator=g, device="cuda:1", dtype=torch.bfloat16)
w2 = torch.randn(E, H, I, generator=g, device="cuda:1", dtype=torch.bfloat16)
s13, s2 = shuffle(w13), shuffle(w2)
print("shuffle permutes values:", not torch.equal(w13, s13))
print("w13 round-trip exact:", torch.equal(unshuffle(s13), w13))
print("w2  round-trip exact:", torch.equal(unshuffle(s2), w2))

# also: unshuffle twice is not identity but unshuffle(shuffle(x)) must be x
print("unshuffle(shuffle) on random:", torch.equal(unshuffle(shuffle(w2)), w2))
print("### DONE", flush=True)
