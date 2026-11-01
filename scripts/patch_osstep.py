#!/usr/bin/env python3
"""mellum.py has no module-level 'os' import, so the step logger's env lookup
raised NameError and killed the model runner."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
if "\nimport os\n" in s:
    print("os already imported")
else:
    old = "from typing import "
    idx = s.index(old)
    s = s[:idx] + "import os\n" + s[idx:]
    p.write_text(s)
    ast.parse(s)
    print("added 'import os' to mellum.py")
old2 = """            if os.environ.get("ATOM_REFERENCE_LOGSTEP"):"""
new2 = """            if _os_step_path():"""
assert old2 in s
s = p.read_text() if "\nimport os\n" in s else s
s = s.replace(old2, new2, 1)
helper = '''
def _os_step_path():
    import os as _os

    return _os.environ.get("ATOM_REFERENCE_LOGSTEP")

'''
anchor = "def _hf_norm("
if anchor in s and "_os_step_path" not in s.split("def _hf_norm")[0]:
    s = s.replace(anchor, helper + anchor, 1)
p.write_text(s)
ast.parse(s)
print("step logger now resolves the path safely")
