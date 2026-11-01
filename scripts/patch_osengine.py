#!/usr/bin/env python3
"""llm_engine.py has no module-level os import; add it so the prompt capture cannot
raise NameError at request time."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/model_engine/llm_engine.py")
s = p.read_text()
if "\nimport os\n" in s:
    print("already imported")
else:
    s = s.replace("import itertools\nimport logging\n", "import itertools\nimport logging\nimport os\n", 1)
    p.write_text(s)
    ast.parse(s)
    print("import os added")
