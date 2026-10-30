#!/usr/bin/env python3
"""Point the gate harness at the real correctness directory."""
from pathlib import Path
p = Path("/workspace/ATOM-tritonmoe/scripts/teacher_forced_gate.py")
s = p.read_text()
old = 'default=str(SHARED / "correctness" / "hf_scenario_c_first5_eager.jsonl")'
new = 'default="/workspace/correctness/hf_scenario_c_first5_eager.jsonl"'
assert old in s
p.write_text(s.replace(old, new, 1))
print("reference path fixed")
