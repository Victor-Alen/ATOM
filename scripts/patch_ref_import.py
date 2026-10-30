#!/usr/bin/env python3
"""Load the HF reference's modeling module as part of a package so its relative
imports work."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """    ref_path = Path(__file__).resolve().parents[2] / "reference" / "hf_mellum" / "modeling_mellum.py"
    spec = importlib.util.spec_from_file_location("_hf_mellum_for_rope", ref_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hf_mellum_for_rope"] = mod
    spec.loader.exec_module(mod)"""
new = """    import types

    ref_dir = Path(__file__).resolve().parents[2] / "reference" / "hf_mellum"
    pkg_name = "_hf_mellum_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(ref_dir)]
        sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(
        pkg_name + ".modeling_mellum", ref_dir / "modeling_mellum.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name + ".modeling_mellum"] = mod
    spec.loader.exec_module(mod)"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("package import fixed")
