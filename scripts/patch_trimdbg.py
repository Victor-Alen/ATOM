#!/usr/bin/env python3
"""Print why the prompt trim does or does not fire."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if _fp == 0 and _PROMPT["ids"]:
            _p = _PROMPT["ids"]"""
new = """        if _fp == 0 and os.environ.get("ATOM_REFERENCE_DEBUG") not in (None, "0"):
            _pv = _PROMPT.get("ids")
            print(
                f"[ref-trim] file_ids={None if not _pv else len(_pv)} "
                f"batch_ids={len(new_ids)} "
                f"file_head={None if not _pv else _pv[:3]} batch_head={new_ids[:3]} "
                f"match={bool(_pv) and new_ids[: len(_pv)] == _pv}",
                flush=True,
            )
        if _fp == 0 and _PROMPT["ids"]:
            _p = _PROMPT["ids"]"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("trim debug added")
