#!/usr/bin/env python3
"""Add a 'none' window rule so we can test whether HF eager ignores sliding windows."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe.py")
s = p.read_text()
old = 'choices=["lt", "le"]'
new = 'choices=["lt", "le", "none"]'
assert old in s
s = s.replace(old, new, 1)
old2 = """            if attn.sliding_window is not None:
                if args.window_rule == "lt":"""
new2 = """            if attn.sliding_window is not None and args.window_rule != "none":
                if args.window_rule == "lt":"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
import ast
ast.parse(s)
print("window-rule none added")
