#!/usr/bin/env python3
"""Debug: compare my slot computation with ATOM's own slot_mapping."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """            write_slots = bs_bt[s, new_p // block_size].long() * block_size + (new_p % block_size)"""
new = """            write_slots = bs_bt[s, new_p // block_size].long() * block_size + (new_p % block_size)
            if os.environ.get("ATOM_REFERENCE_DEBUG") in ("1", "2") and li == 0 and s == 0:
                sm = getattr(md, "slot_mapping", None)
                smv = None if sm is None else sm[lo:hi].tolist()[:8]
                print(f"[ref] slots_mine={write_slots.tolist()[:8]} slots_atom={smv} bt={bs_bt[s].tolist()[:6]} bs={block_size}", flush=True)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("slot debug added")
