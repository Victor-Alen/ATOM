#!/usr/bin/env python3
"""Add per-forward diagnostics and a per-step logits dump."""
import ast
from pathlib import Path

# --- 1. richer forward diagnostics -----------------------------------------
p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """                if os.environ.get("ATOM_REFERENCE_CHUNKLOG") == "1":
                    print(
                        f"[chunk] n={_n} bs={bs} pos={_CHUNK['pos']} ctx={ctx} "
                        f"slots={int(len(ctx_slots[s]))}",
                        flush=True,
                    )"""
new = """                if os.environ.get("ATOM_REFERENCE_CHUNKLOG") == "1":
                    _ws = write_slots[:3].tolist() if hasattr(write_slots, "tolist") else write_slots
                    _cs = ctx_slots[s]
                    _cs_head = _cs[:3].tolist() if hasattr(_cs, "tolist") else _cs
                    _cs_tail = _cs[-2:].tolist() if hasattr(_cs, "tolist") else _cs
                    print(
                        f"[chunk] fwd={_n} q_len={int(len(new_pos))} ctx={ctx} "
                        f"pos={_CHUNK['pos']} ctx_slots_len={int(len(_cs))} "
                        f"ctx_head={_cs_head} ctx_tail={_cs_tail} write={_ws}",
                        flush=True,
                    )"""
assert old in s
s = s.replace(old, new, 1)

old2 = """    _HIST.append(list(new_ids))"""
if old2 in s:
    s = s.replace(
        old2,
        """    if os.environ.get("ATOM_REFERENCE_CHUNKLOG") == "1":
        print(f"[hist] len_before={len(_HIST)} new={len(list(new_ids))}", flush=True)
    _HIST.append(list(new_ids))""",
        1,
    )
p.write_text(s)
ast.parse(s)
print("reference_mode diagnostics added")

# --- 2. per-step logits dump ----------------------------------------------
q = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
t = q.read_text()
anchor = "        if _ref_path:"
idx = t.find(anchor)
assert idx != -1, "compute_logits anchor missing"
inject = '''        if _ref_path:
            import os as _os
            _ld = _os.environ.get("ATOM_REFERENCE_LOGITDUMP")
            if _ld:
                try:
                    _k = len(_os.listdir(_ld))
                    torch.save(out[0].detach().float().cpu(), _os.path.join(_ld, f"logits_{_k:02d}.pt"))
                except Exception as _e:  # noqa: BLE001
                    print("[logitdump]", _e, flush=True)
'''
t = t[:idx] + inject + t[idx:]
q.write_text(t)
ast.parse(t)
print("logits dump added")
