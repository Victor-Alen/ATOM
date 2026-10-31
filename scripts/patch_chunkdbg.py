#!/usr/bin/env python3
"""Log the per-call chunk pattern (n_new, first position, history length) so the
12580-token chunk boundary can be inspected."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        new_ids = input_ids.tolist()
        hist = history_after(new_ids, int(positions[0].item()) if positions.numel() else 0)"""
new = """        new_ids = input_ids.tolist()
        _fp = int(positions[0].item()) if positions.numel() else 0
        hist = history_after(new_ids, _fp)
        if os.environ.get("ATOM_REFERENCE_DEBUG") not in (None, "0"):
            print(
                f"[ref-chunk] n_new={len(new_ids)} first_pos={_fp} hist={len(hist)}",
                flush=True,
            )"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("chunk debug added")
