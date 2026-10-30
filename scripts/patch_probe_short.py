#!/usr/bin/env python3
"""Shorten the probe prompt so the full-sequence forward fits in GPU memory."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe.py")
s = p.read_text()
old = '    prompt_ids = rec["prompt_token_ids"]'
new = '    prompt_ids = rec["prompt_token_ids"][: args.prompt_tokens]'
assert old in s
s = s.replace(old, new, 1)
old2 = '    ap.add_argument("--window-rule", default="lt", choices=["lt", "le"])'
new2 = (old2 + chr(10) + '    ap.add_argument("--prompt-tokens", type=int, default=512)')
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
import ast
ast.parse(s)
print("probe now truncates the prompt to --prompt-tokens (default 512)")
