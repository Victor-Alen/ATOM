#!/usr/bin/env python3
"""Golden script: optional --extra (append the first reference token)."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/golden_step2.py")
s = p.read_text()
if "add_argument" not in s:
    s = s.replace(
        '    ref = load_reference_module()',
        '    import argparse' + chr(10) +
        '    ap = argparse.ArgumentParser()' + chr(10) +
        '    ap.add_argument("--extra", type=int, default=0)' + chr(10) +
        '    ap.add_argument("--tokens", type=int, default=0)' + chr(10) +
        '    args_in = ap.parse_args()' + chr(10) +
        '    ref = load_reference_module()', 1)
    s = s.replace(
        '    ids = torch.tensor([rec["prompt_token_ids"] + [rec["token_ids"][0]]], dtype=torch.long, device="cuda:0")',
        '    seq = list(rec["prompt_token_ids"])' + chr(10) +
        '    if args_in.tokens:' + chr(10) +
        '        seq = seq[: args_in.tokens]' + chr(10) +
        '    if args_in.extra:' + chr(10) +
        '        seq = seq + list(rec["token_ids"][: args_in.extra])' + chr(10) +
        '    ids = torch.tensor([seq], dtype=torch.long, device="cuda:0")', 1)
    p.write_text(s)
    import ast
    ast.parse(s)
    print("golden accepts --tokens/--extra")
