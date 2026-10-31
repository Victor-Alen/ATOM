#!/usr/bin/env python3
"""The KV *write* also indexed the engine block table, so a decode step (where
bs_tables is None) raised TypeError and killed the runner. In recompute mode the
reference path owns the buffer, so the write slots are simply the token positions."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = "            write_slots = bs_bt[s, new_pos // block_size].long() * block_size + (new_pos % block_size)"
new = """            if _recomp:
                write_slots = new_pos
            else:
                write_slots = (
                    bs_bt[s, new_pos // block_size].long() * block_size + (new_pos % block_size)
                )"""
assert old in s
s = s.replace(old, new, 1)

# also make the debug line safe with bs_bt=None
old2 = """                print(f"[ref] slots_mine={write_slots.tolist()[:8]} slots_atom={smv} bt={bs_bt[s].tolist()[:6]} bs={block_size}", flush=True)"""
new2 = """                bt_dbg = None if bs_bt is None else bs_bt[s].tolist()[:6]
                print(f"[ref] slots_mine={write_slots.tolist()[:8]} slots_atom={smv} bt={bt_dbg} bs={block_size}", flush=True)"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("write slots fixed for block_tables=None")
