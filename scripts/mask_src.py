#!/usr/bin/env python3
"""Unit test: my sliding-window mask vs HF's create_sliding_window_causal_mask."""
import inspect
import sys

import torch

sys.path.insert(0, "/workspace/ATOM-tritonmoe")
from transformers.masking_utils import create_sliding_window_causal_mask  # noqa: E402

print(inspect.signature(create_sliding_window_causal_mask), flush=True)

S = 2048
W = 1024
qi = torch.arange(S).view(-1, 1)
ki = torch.arange(S).view(1, -1)
mine = (ki <= qi) & ((qi - ki) < W)

# try to reproduce HF's mask through its own utility
import types
from transformers.masking_utils import sdpa_mask  # noqa: E402

try:
    src = inspect.getsource(create_sliding_window_causal_mask)
    print("--- HF sliding mask source (first 25 lines) ---", flush=True)
    print("\n".join(src.splitlines()[:25]), flush=True)
except Exception as exc:  # noqa: BLE001
    print("source unavailable:", exc, flush=True)
print("### MASK_SRC_DONE", flush=True)
