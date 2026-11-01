#!/usr/bin/env python3
"""The scheduler can attach the previous request's last sampled token to the tail of a
new prefill (observed: 1470-token prompt scheduled as 1471 with the old prediction last).
Track our own last prediction and drop that trailing token when it matches."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
s = s.replace("_HIST: list = []", "_HIST: list = []\n_LAST_PRED = [None]", 1)

old = """        new_ids = input_ids.tolist()
        _fp = int(positions[0].item()) if positions.numel() else 0"""
new = """        new_ids = input_ids.tolist()
        _fp = int(positions[0].item()) if positions.numel() else 0
        if (
            _fp == 0
            and len(new_ids) > 1
            and _LAST_PRED[0] is not None
            and new_ids[-1] == _LAST_PRED[0]
        ):
            new_ids = new_ids[:-1]
            input_ids = input_ids[:-1]
            if positions.numel() > len(new_ids):
                positions = positions[: len(new_ids)]
            if os.environ.get("ATOM_REFERENCE_DEBUG") not in (None, "0"):
                print("[ref] dropped leaked deferred token", flush=True)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)

p2 = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s2 = p2.read_text()
old2 = """            _dmp = None"""
new2 = """            try:
                import atom.models.reference_mode as _rm

                _rm._LAST_PRED[0] = int(_logits[-1].argmax().item())
            except Exception:  # noqa: BLE001
                pass
            _dmp = None"""
assert old2 in s2
s2 = s2.replace(old2, new2, 1)
p2.write_text(s2)
ast.parse(s2)
print("leak guard installed")
