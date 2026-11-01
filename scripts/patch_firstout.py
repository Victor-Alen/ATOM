#!/usr/bin/env python3
"""The engine appends the previous request's FIRST sampled token to the tail of a new
prefill (confirmed on three different prompts). Track that token and trim it."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
s = s.replace('_PROMPT = {"ids": None}', '_PROMPT = {"ids": None}\n_FIRST_OUT = [None]\n_PENDING_FIRST = [False]', 1)

old = """        if _fp == 0 and os.environ.get("ATOM_REFERENCE_DEBUG") not in (None, "0"):
            _pv = _PROMPT.get("ids")
            print(
                f"[ref-trim] file_ids={None if not _pv else len(_pv)} "
                f"batch_ids={len(new_ids)} "
                f"file_head={None if not _pv else _pv[:3]} batch_head={new_ids[:3]} "
                f"match={bool(_pv) and new_ids[: len(_pv)] == _pv}",
                flush=True,
            )
        if _fp == 0 and _PROMPT["ids"]:
            _p = _PROMPT["ids"]
            if len(_p) <= len(new_ids) and new_ids[: len(_p)] == _p:
                if len(_p) < len(new_ids):
                    if os.environ.get("ATOM_REFERENCE_DEBUG") not in (None, "0"):
                        print(
                            f"[ref] trimmed input_ids {len(new_ids)} -> {len(_p)} "
                            "(engine appended a token)",
                            flush=True,
                        )
                    new_ids = new_ids[: len(_p)]
                    input_ids = input_ids[: len(_p)]
                    if positions.numel() > len(new_ids):
                        positions = positions[: len(new_ids)]
                _PROMPT["ids"] = None"""
new = """        if (
            _fp == 0
            and len(new_ids) > 1
            and _FIRST_OUT[0] is not None
            and new_ids[-1] == _FIRST_OUT[0]
        ):
            new_ids = new_ids[:-1]
            input_ids = input_ids[:-1]
            if positions.numel() > len(new_ids):
                positions = positions[: len(new_ids)]
            if os.environ.get("ATOM_REFERENCE_DEBUG") not in (None, "0"):
                print(
                    f"[ref] trimmed engine-appended token {_FIRST_OUT[0]} "
                    f"({len(new_ids) + 1} -> {len(new_ids)} tokens)",
                    flush=True,
                )
        if _fp == 0:
            _PENDING_FIRST[0] = True"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)

p2 = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s2 = p2.read_text()
old2 = """            try:
                import atom.models.reference_mode as _rm

                _rm._LAST_PRED[0] = int(_logits[-1].argmax().item())
            except Exception:  # noqa: BLE001
                pass"""
new2 = """            try:
                import atom.models.reference_mode as _rm

                _pred = int(_logits[-1].argmax().item())
                _rm._LAST_PRED[0] = _pred
                if _rm._PENDING_FIRST[0]:
                    _rm._FIRST_OUT[0] = _pred
                    _rm._PENDING_FIRST[0] = False
            except Exception:  # noqa: BLE001
                pass"""
assert old2 in s2
s2 = s2.replace(old2, new2, 1)
p2.write_text(s2)
ast.parse(s2)
print("first-output tracking installed")
