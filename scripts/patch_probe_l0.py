#!/usr/bin/env python3
"""Two changes: allow disabling the allocator-fraction lift (it was a workaround for
tight VRAM) and add debug probes inside layer 0 so a hang can be localized."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """            _frac = float(os.environ.get("ATOM_REFERENCE_MEM_FRACTION", "0.85"))
            torch.cuda.set_per_process_memory_fraction(_frac, input_ids.device.index or 0)
            print(f"[ref] allocator fraction lifted to {_frac}", flush=True)"""
new = """            _frac = float(os.environ.get("ATOM_REFERENCE_MEM_FRACTION", "0.85"))
            if _frac > 0:
                torch.cuda.set_per_process_memory_fraction(_frac, input_ids.device.index or 0)
                print(f"[ref] allocator fraction lifted to {_frac}", flush=True)
            else:
                print("[ref] allocator fraction unchanged (disabled)", flush=True)"""
assert old in s
s = s.replace(old, new, 1)

probe = '''        if os.environ.get("ATOM_REFERENCE_DEBUG") == "2":
            print(f"[ref] L{li} start x={tuple(x.shape)}", flush=True)
'''
anchor = """        for s in range(bs):
            lo, hi = cu_q[s], cu_q[s + 1]
            h = _hf_norm(x[lo:hi], layer.input_layernorm.weight, eps)"""
assert anchor in s
s = s.replace(anchor, probe + anchor, 1)

probe2 = '''            if os.environ.get("ATOM_REFERENCE_DEBUG") == "2":
                print(f"[ref] L{li} proj q={tuple(q.shape)}", flush=True)
'''
anchor2 = "            cos, sin = rot(li, pos_chunk, q.dtype) if False else rot(li, pos[lo:hi], q.dtype)"
if anchor2 in s:
    s = s.replace(anchor2, anchor2 + "\n" + probe2, 1)
else:
    # fall back: insert before the attention chunk loop
    anchor3 = "            for c0 in range(0, n_new, CH):"
    if anchor3 in s:
        s = s.replace(anchor3, probe2 + anchor3, 1)
p.write_text(s)
ast.parse(s)
print("fraction switch + layer0 probes installed")
