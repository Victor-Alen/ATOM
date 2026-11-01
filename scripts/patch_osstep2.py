#!/usr/bin/env python3
"""Make the step logger self-contained: the module-level helper landed in the wrong
scope (compute_logits is nested), so resolve the path inline."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = """            if _os_step_path():"""
new = """            _lsp = None
            try:
                import os as _os2

                _lsp = _os2.environ.get("ATOM_REFERENCE_LOGSTEP")
            except Exception:  # noqa: BLE001
                _lsp = None
            if _lsp:"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace("""                    _path = os.environ["ATOM_REFERENCE_LOGSTEP"]""",
              """                    _path = _lsp""", 1)
p.write_text(s)
ast.parse(s)
print("step logger is self-contained")
