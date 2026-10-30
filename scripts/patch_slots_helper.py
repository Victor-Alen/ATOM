#!/usr/bin/env python3
"""Extract the KV-slot derivation into a testable helper (the block-table fallback
fix deserves a unit test, not just an end-to-end claim)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

helper = '''
def seq_slots(positions, cu_q, ctx_lens, bs_bt, block_size, recomp, device):
    """KV slots for every request in the batch.

    In recompute mode the reference path owns the KV buffer, so slots are simply
    0..ctx-1 and the engine block table is not consulted at all (it is None on
    decode steps, which is what used to disable the reference path).
    """
    slots = []
    for s in range(len(ctx_lens)):
        if recomp:
            slots.append(torch.arange(int(ctx_lens[s]), device=device))
            continue
        p = positions[cu_q[s] : cu_q[s + 1]].long()
        n_ctx = int(p.max().item()) + 1
        q = torch.arange(n_ctx, device=device)
        slots.append(bs_bt[s, q // block_size].long() * block_size + (q % block_size))
    return slots


'''
anchor = "def reference_forward(model, input_ids, positions):"
assert anchor in s
s = s.replace(anchor, helper.lstrip("\n") + anchor, 1)

old = """    ctx_slots = []
    for s in range(bs):
        if _recomp:
            ctx_slots.append(torch.arange(int(ctx_lens[s]), device=dev))
            continue
        p = positions[cu_q[s]:cu_q[s + 1]].long()
        n_ctx = int(p.max().item()) + 1
        q = torch.arange(n_ctx, device=dev)
        ctx_slots.append(bs_bt[s, q // block_size].long() * block_size + (q % block_size))"""
new = "    ctx_slots = seq_slots(positions, cu_q, ctx_lens, bs_bt, block_size, _recomp, dev)"
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("seq_slots extracted")
