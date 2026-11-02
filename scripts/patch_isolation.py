#!/usr/bin/env python3
"""Isolate per-request reference state without relying on first_pos.

In recompute mode every decode step also starts at position 0, so first_pos cannot
identify a new request. Instead compare the incoming token stream with the stored
history: a continuation either equals it or extends it; anything else is a new request.
"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = '''def history_after(new_ids, first_pos=0):
    """Update the single-sequence token history and return it.

    A request starts at position 0 (prefill, possibly split into several chunks);
    everything else continues the current sequence. Using 'multi-token means a new
    request' was wrong because chunked prefill sends several multi-token chunks.
    """
    if int(first_pos) == 0:
        _HIST[:] = list(new_ids)
    else:
        _HIST.extend(list(new_ids))
    return _HIST'''

new = '''def _reset_reference_state():
    """Drop every piece of per-request state in the reference path."""
    global _PENDING_FIRST, _LAST_PRED
    _HIST.clear()
    _KV.clear()
    _WRITTEN.clear()
    _PENDING_FIRST = None
    _LAST_PRED = None


def history_after(new_ids, first_pos=0):
    """Update the single-sequence token history and return it.

    A decode step in recompute mode replays the whole history from position 0, so the
    absolute position cannot distinguish a new request from a replayed one. The
    incoming stream is classified instead: it either equals the stored history, or it
    extends it, or it is something else - which means a new request arrived and all
    per-request state must be reset.
    """
    ids = list(new_ids)
    prev = _HIST
    n = len(ids)
    same = n <= len(prev) and prev[:n] == ids
    extends = n > len(prev) and ids[: len(prev)] == prev
    if same:
        return _HIST
    if extends:
        _HIST[:] = ids
        return _HIST
    _reset_reference_state()
    _HIST[:] = ids
    return _HIST'''

assert old in s, "history_after not found"
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("request isolation added")
