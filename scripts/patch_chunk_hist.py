#!/usr/bin/env python3
"""Chunked prefill: consecutive chunks of the same prompt are multi-token calls, so
'multi-token means a new request' reset the history for every chunk. Decide by the
first position instead: a chunk that starts at position 0 begins a new request."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """def history_after(new_ids):
    \"\"\"Update the single-sequence token history and return it.

    A multi-token call is a prefill and replaces the history; a single-token call
    is a decode and appends. Keying this on the engine block table was wrong
    because decode steps have no block table (bs_bt is None).
    \"\"\"
    if len(new_ids) > 1:
        _HIST[:] = list(new_ids)
    else:
        _HIST.append(new_ids[0])
    return _HIST"""
new = """def history_after(new_ids, first_pos=0):
    \"\"\"Update the single-sequence token history and return it.

    A request starts at position 0 (prefill, possibly split into several chunks);
    everything else continues the current sequence. Using 'multi-token means a new
    request' was wrong because chunked prefill sends several multi-token chunks.
    \"\"\"
    if int(first_pos) == 0:
        _HIST[:] = list(new_ids)
    else:
        _HIST.extend(list(new_ids))
    return _HIST"""
assert old in s
s = s.replace(old, new, 1)

old2 = """        new_ids = input_ids.tolist()
        hist = history_after(new_ids)"""
new2 = """        new_ids = input_ids.tolist()
        hist = history_after(new_ids, int(positions[0].item()) if positions.numel() else 0)"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("history now keyed on the first position (chunk-safe)")
