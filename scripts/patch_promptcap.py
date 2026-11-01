#!/usr/bin/env python3
"""Give the reference path the authoritative prompt: capture the tokenized ids at the
HTTP layer and use them for the next prefill instead of the engine's padded/scheduled
buffer (which can carry the previous request's token)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/model_engine/llm_engine.py")
s = p.read_text()
old = """        tokens = (
            self.tokenizer.encode(prompt_or_tokens)
            if isinstance(prompt_or_tokens, str)
            else prompt_or_tokens
        )"""
new = """        tokens = (
            self.tokenizer.encode(prompt_or_tokens)
            if isinstance(prompt_or_tokens, str)
            else prompt_or_tokens
        )
        if os.environ.get("ATOM_REFERENCE_MODE") == "1":
            try:
                import atom.models.reference_mode as _rm

                _rm._PROMPT["ids"] = [int(t) for t in tokens]
            except Exception:  # noqa: BLE001
                pass"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)

p2 = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s2 = p2.read_text()
s2 = s2.replace("_LAST_PRED = [None]", "_LAST_PRED = [None]\n_PROMPT = {\"ids\": None}", 1)
old2 = """        if (
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
new2 = """        if _fp == 0 and _PROMPT["ids"]:
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
assert old2 in s2
s2 = s2.replace(old2, new2, 1)
p2.write_text(s2)
ast.parse(s2)
print("authoritative prompt capture installed")
