#!/usr/bin/env python3
"""The HTTP layer and the model runner are separate processes, so the captured prompt
must travel through a file."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/model_engine/llm_engine.py")
s = p.read_text()
old = """        if os.environ.get("ATOM_REFERENCE_MODE") == "1":
            try:
                import atom.models.reference_mode as _rm

                _rm._PROMPT["ids"] = [int(t) for t in tokens]
            except Exception:  # noqa: BLE001
                pass"""
new = """        if os.environ.get("ATOM_REFERENCE_MODE") == "1":
            try:
                import json as _json

                _pf = os.environ.get(
                    "ATOM_REFERENCE_PROMPT_FILE", "/tmp/atom_reference_prompt.json"
                )
                with open(_pf, "w", encoding="utf-8") as _fh:
                    _json.dump({"ids": [int(t) for t in tokens]}, _fh)
            except Exception:  # noqa: BLE001
                pass"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)

p2 = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s2 = p2.read_text()
old2 = """        if _fp == 0 and _PROMPT["ids"]:
            _p = _PROMPT["ids"]"""
new2 = """        if _fp == 0 and _PROMPT["ids"] is None:
            try:
                import json as _json

                _pf = os.environ.get(
                    "ATOM_REFERENCE_PROMPT_FILE", "/tmp/atom_reference_prompt.json"
                )
                with open(_pf, encoding="utf-8") as _fh:
                    _PROMPT["ids"] = _json.load(_fh)["ids"]
            except Exception:  # noqa: BLE001
                _PROMPT["ids"] = []
        if _fp == 0 and _PROMPT["ids"]:
            _p = _PROMPT["ids"]"""
assert old2 in s2
s2 = s2.replace(old2, new2, 1)
p2.write_text(s2)
ast.parse(s2)
print("prompt now travels via file")
