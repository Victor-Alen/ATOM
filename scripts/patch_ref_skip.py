#!/usr/bin/env python3
"""Skip the reference path during model warmup, and add per-layer debug output."""
import ast
from pathlib import Path

r = Path("/workspace/ATOM-tritonmoe/atom/model_engine/model_runner.py")
s = r.read_text()
if "ATOM_REFERENCE_SKIP" not in s:
    old = "        self.warmup_model()"
    new = ('        __import__("os").environ["ATOM_REFERENCE_SKIP"] = "1"\n'
           "        try:\n"
           "            self.warmup_model()\n"
           "        finally:\n"
           '            __import__("os").environ.pop("ATOM_REFERENCE_SKIP", None)')
    assert old in s, "warmup call site not found"
    s = s.replace(old, new, 1)
    r.write_text(s)
    ast.parse(s)
    print("warmup skip wired")

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
t = p.read_text()
if "ATOM_REFERENCE_SKIP" not in t:
    old2 = "    fwd_ctx = get_forward_context()"
    new2 = ('    fwd_ctx = get_forward_context()\n'
            '    if os.environ.get("ATOM_REFERENCE_SKIP") == "1":\n'
            "        return None")
    assert old2 in t
    t = t.replace(old2, new2, 1)
if "ATOM_REFERENCE_DEBUG == \"2\"" not in t:
    old3 = "        x = x + moe\n\n    return _hf_norm(x, model.model.norm.weight, eps)"
    new3 = ('        x = x + moe\n'
            '        if os.environ.get("ATOM_REFERENCE_DEBUG") == "2" and li < 3:\n'
            '            print(f"[ref] layer {li} done shape={tuple(x.shape)} "'
            'f"absmax={float(x.abs().max()):.4f}", flush=True)\n\n'
            "    return _hf_norm(x, model.model.norm.weight, eps)")
    assert old3 in t
    t = t.replace(old3, new3, 1)
p.write_text(t)
ast.parse(t)
print("reference_mode debug + skip wired")
