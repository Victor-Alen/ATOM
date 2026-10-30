#!/usr/bin/env python3
"""Ensure os is imported at module top and the stray late import is removed."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
s = s.replace("import os  # noqa: E402  (kept last so the module-level constant above can use it)\n\n\n", "")
s = s.replace("import os  # noqa: E402  (kept last so the module-level constant above can use it)", "")
if "\nimport os\n" not in s.split("import importlib")[0]:
    s = "import os\n" + s
p.write_text(s)
import ast
ast.parse(s)
print("top-level import block:")
print(chr(10).join(s.splitlines()[:8]))
