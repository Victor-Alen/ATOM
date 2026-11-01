#!/usr/bin/env python3
"""Use the prompt captured at the HTTP layer, with a prefix check and one-shot delete,
so a leaked trailing token can never reach the forward."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """        if _fp == 0 and _PROMPT["ids"] is None:
            try:
                import json as _json

                _pf = os.environ.get(
                    "ATOM_REFERENCE_PROMPT_FILE", "/tmp/atom_reference_prompt.json"
                )
                with open(_pf, encoding="utf-8") as _fh:
                    _PROMPT["ids"] = _json.load(_fh)["ids"]
            except Exception:  # noqa: BLE001
                _PROMPT["ids"] = []"""
new = """        if _fp == 0 and _PROMPT["ids"] is None:
            try:
                import json as _json
                from pathlib import Path as _P

                _pf = os.environ.get(
                    "ATOM_REFERENCE_PROMPT_FILE", "/tmp/atom_reference_prompt.json"
                )
                _pp = _P(_pf)
                _PROMPT["ids"] = _json.loads(_pp.read_text(encoding="utf-8"))["ids"]
                _pp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                _PROMPT["ids"] = []"""
assert old in s
s = s.replace(old, new, 1)
old2 = """        if (
            _fp == 0
            and len(new_ids) > 1
            and _FIRST_OUT[0] is not None
            and new_ids[-1] == _FIRST_OUT[0]
        ):"""
new2 = """        _pfile = _PROMPT["ids"] if _fp == 0 else None
        if (
            _pfile
            and len(_pfile) < len(new_ids)
            and new_ids[: len(_pfile)] == _pfile
        ):
            new_ids = new_ids[: len(_pfile)]
            input_ids = input_ids[: len(_pfile)]
            if positions.numel() > len(new_ids):
                positions = positions[: len(new_ids)]
            if os.environ.get("ATOM_REFERENCE_DEBUG") not in (None, "0"):
                print(
                    f"[ref] trimmed input_ids to captured prompt {len(_pfile)} "
                    f"(dropped {len(_pfile) and 1})",
                    flush=True,
                )
        if (
            _fp == 0
            and len(new_ids) > 1
            and _FIRST_OUT[0] is not None
            and new_ids[-1] == _FIRST_OUT[0]
        ):"""
assert old2 in s
s = s.replace(old2, new2, 1)
s = s.replace('            _PROMPT["ids"] = None', '            _PROMPT["ids"] = None', 1)
p.write_text(s)
ast.parse(s)
print("prompt-file trim with prefix check installed")
