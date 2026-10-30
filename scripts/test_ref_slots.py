#!/usr/bin/env python3
"""Unit test for the reference path's KV-slot derivation."""
import sys

import torch

sys.path.insert(0, "/workspace/ATOM-tritonmoe")
from atom.models.reference_mode import seq_slots  # noqa: E402

BS = 16

# 1) recompute mode with no block table (the decode-step case that used to bail out)
slots = seq_slots(None, [0, 5], [5], None, BS, True, "cpu")
assert slots[0].tolist() == [0, 1, 2, 3, 4], slots[0].tolist()
print("recompute + block_tables=None -> sequential slots OK")

# 2) recompute mode on a longer history
slots = seq_slots(None, [0, 3], [3], None, BS, True, "cpu")
assert slots[0].tolist() == [0, 1, 2]
print("recompute + short history OK")

# 3) normal mode: block table mapping must match the engine's own layout
block_table = torch.tensor([[7, 2, 9]])
positions = torch.tensor([0, 1, 2, 3, 4, 5])
slots = seq_slots(positions, [0, 6], [6], block_table, BS, False, "cpu")
expected = []
for p in range(6):
    blk = block_table[0, p // BS].item()
    expected.append(blk * BS + p % BS)
assert slots[0].tolist() == expected, (slots[0].tolist(), expected)
print("block-table mapping OK:", slots[0].tolist())

# 4) two requests in one batch
positions = torch.tensor([0, 1, 0, 1])
slots = seq_slots(positions, [0, 2, 4], [2, 2], torch.tensor([[1, 0], [4, 0]]), BS, False, "cpu")
assert slots[0].tolist() == [16, 17]
assert slots[1].tolist() == [64, 65]
print("two-request batch OK")
print("### ALL PASS")
