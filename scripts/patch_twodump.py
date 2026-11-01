#!/usr/bin/env python3
"""Dump the first two prefills separately so in-engine run-to-run determinism can be
tested on the same input."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
s = s.replace(
    '    _dump_call = bool(_dump_dir) and "layers" not in _DUMPED and int(x.shape[0]) > 1',
    '    _dump_ok = bool(_dump_dir) and int(x.shape[0]) > 1\n'
    '    _dump_n = len([k for k in _DUMPED if k.startswith("layers")])\n'
    '    _dump_call = _dump_ok and _dump_n < 2',
    1,
)
s = s.replace(
    """    if _dump_call:
        _DUMPED.add("layers")
        _LDUMP["final"] = out[-1].detach().float().cpu()
        torch.save(_LDUMP, f"{_dump_dir}/layers.pt")""",
    """    if _dump_call:
        _DUMPED.add(f"layers{_dump_n}")
        _LDUMP["final"] = out[-1].detach().float().cpu()
        torch.save(_LDUMP, f"{_dump_dir}/layers_{_dump_n}.pt")""",
    1,
)
p.write_text(s)
ast.parse(s)
print("two-prefill dump armed")
