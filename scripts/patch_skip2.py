#!/usr/bin/env python3
"""Move the warmup skip into ModelRunner.warmup_model (the process that runs it)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/model_engine/model_runner.py")
s = p.read_text()

revert = ('        __import__("os").environ["ATOM_REFERENCE_SKIP"] = "1"\n'
          "        try:\n"
          "            self.warmup_model()\n"
          "        finally:\n"
          '            __import__("os").environ.pop("ATOM_REFERENCE_SKIP", None)')
if revert in s:
    s = s.replace(revert, "        self.warmup_model()", 1)
    print("reverted call-site patch")

old = "    def warmup_model(self):\n        start_time = time.time()"
new = ('    def warmup_model(self):\n'
       '        __import__("os").environ["ATOM_REFERENCE_SKIP"] = "1"\n'
       "        start_time = time.time()")
assert old in s
s = s.replace(old, new, 1)

old2 = "        self.forward(dummy_batch)\n        self.tokenID_processor.clean()"
new2 = ('        self.forward(dummy_batch)\n'
        '        __import__("os").environ.pop("ATOM_REFERENCE_SKIP", None)\n'
        "        self.tokenID_processor.clean()")
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("warmup skip now inside warmup_model")
