#!/usr/bin/env python3
"""The reference path needs more allocator headroom than the fast path: lift
ATOM's per-process memory fraction once, otherwise PyTorch aborts at ~util bytes."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    if os.environ.get("ATOM_REFERENCE_SKIP") == "1":
        return None"""
new = """    if os.environ.get("ATOM_REFERENCE_SKIP") == "1":
        return None
    if "frac" not in _ROT:
        try:
            _frac = float(os.environ.get("ATOM_REFERENCE_MEM_FRACTION", "0.85"))
            torch.cuda.set_per_process_memory_fraction(_frac, input_ids.device.index or 0)
            print(f"[ref] allocator fraction lifted to {_frac}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[ref] could not lift allocator fraction: {exc}", flush=True)
        _ROT["frac"] = True"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("allocator fraction lift added")
