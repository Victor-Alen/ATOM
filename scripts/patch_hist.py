#!/usr/bin/env python3
"""The recompute history must not be keyed on the engine block table: the decode
step arrives with block_tables=None, so keying on bs_bt[0,0] split the history in
two (prefill under one key, decode under key 0) and a decode could recompute from
a single token. Use one module-level history plus a testable helper."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

if "def history_after" not in s:
    helper = '''_HIST: list = []


def history_after(new_ids):
    """Update the single-sequence token history and return it.

    A multi-token call is a prefill and replaces the history; a single-token call
    is a decode and appends. Keying this on the engine block table was wrong
    because decode steps have no block table (bs_bt is None).
    """
    if len(new_ids) > 1:
        _HIST[:] = list(new_ids)
    else:
        _HIST.append(new_ids[0])
    return _HIST


'''
    anchor = "def seq_slots("
    assert anchor in s
    s = s.replace(anchor, helper + anchor, 1)

old = """        key = int(bs_bt[0, 0].item()) if bs_bt is not None else 0
        hist = _TOKENS.setdefault(key, [])
        new_ids = input_ids.tolist()
        if len(new_ids) > 1:
            hist[:] = new_ids
        else:
            hist.append(new_ids[0])"""
new = """        new_ids = input_ids.tolist()
        hist = history_after(new_ids)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("history helper installed")
