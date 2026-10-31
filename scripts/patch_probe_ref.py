#!/usr/bin/env python3
"""Let the probe take an explicit reference jsonl (A-0002 lives in the A file)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/scripts/ref_arith_probe3.py")
s = p.read_text()
if "--ref-file" not in s:
    s = s.replace('    ap.add_argument("--prompt-id"',
                  '    ap.add_argument("--ref-file", default="/workspace/correctness/hf_scenario_c_200_eager.jsonl")\n    ap.add_argument("--prompt-id"', 1)
    s = s.replace('for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):',
                  'for line in open(args.ref_file, encoding="utf-8"):', 1)
    p.write_text(s)
    ast.parse(s)
    print("probe accepts --ref-file")
else:
    print("already patched")
