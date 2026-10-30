#!/usr/bin/env python3
"""Derive KV slots from the token positions themselves instead of context_lens.

Engines differ on whether context_lens counts the tokens of the current step;
positions are unambiguous: slot(p) = block_table[p // bs] * bs + p % bs.
"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """    ctx_slots = []
    for s in range(bs):
        p = torch.arange(ctx_lens[s], device=dev)
        ctx_slots.append(bs_bt[s, p // block_size].long() * block_size + (p % block_size))"""
new = """    ctx_slots = []
    for s in range(bs):
        p = positions[cu_q[s]:cu_q[s + 1]].long()
        n_ctx = int(p.max().item()) + 1
        q = torch.arange(n_ctx, device=dev)
        ctx_slots.append(bs_bt[s, q // block_size].long() * block_size + (q % block_size))"""
assert old in s
s = s.replace(old, new, 1)

old2 = """            ctx = ctx_lens[s]
            n_new = hi - lo
            new_pos = positions[lo:hi].long()
            new_p = torch.arange(ctx - n_new, ctx, device=dev)
            write_slots = bs_bt[s, new_p // block_size].long() * block_size + (new_p % block_size)"""
new2 = """            n_new = hi - lo
            new_pos = positions[lo:hi].long()
            ctx = int(new_pos.max().item()) + 1
            write_slots = bs_bt[s, new_pos // block_size].long() * block_size + (new_pos % block_size)"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("slot derivation now position-based")
