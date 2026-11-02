#!/usr/bin/env python3
"""Check that aiter's expert shuffle is exactly inverted by reference_mode._unshuffle."""
import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/ATOM-tritonmoe")
from atom.models.reference_mode import _unshuffle  # noqa: E402

aiter_ok = True
try:
    import aiter
except Exception as exc:  # noqa: BLE001
    aiter_ok = False
    print("aiter import failed:", exc)

torch.manual_seed(0)
gu = torch.randn(4, 1792, 2304, dtype=torch.bfloat16, device="cuda")
dn = torch.randn(4, 2304, 896, dtype=torch.bfloat16, device="cuda")

print("aiter available:", aiter_ok)
if aiter_ok and hasattr(aiter, "shuffle_weight"):
    sh = aiter.shuffle_weight(gu, layout=(16, 16))
    back = _unshuffle(sh)
    print("aiter shuffle -> _unshuffle: max|delta| =", float((back - gu).abs().max()))
    print("shuffled == original?", bool(torch.equal(sh, gu)))
else:
    print("aiter.shuffle_weight not available; testing _unshuffle idempotence on a permutation")
    back = _unshuffle(gu)
    print("_unshuffle changes tensor?", not bool(torch.equal(back, gu)), flush=True)
print("### SHUF_DONE")
