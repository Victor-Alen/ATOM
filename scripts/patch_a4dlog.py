#!/usr/bin/env python3
"""Log the 4-D attention branch shapes and make decode-step dumps unconditional."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()

old = """                if os.environ.get("ATOM_REFERENCE_ATTN4D") == "1":
                    q4 = qh.reshape(1, nH, qh.shape[2], D)[:, :, c0:c1, :]"""
new = """                if os.environ.get("ATOM_REFERENCE_ATTN4D") == "1":
                    q4 = qh.reshape(1, nH, qh.shape[2], D)[:, :, c0:c1, :]
                    if os.environ.get("ATOM_REFERENCE_ATTN4D_LOG") == "1":
                        print(
                            f"[a4d] L{li} chunk={_CHUNK.get('n')} n_new={n_new} "
                            f"nH={nH} rep={rep} ctx={ctx} rows={c1 - c0} "
                            f"k4e={tuple(kk.shape)}",
                            flush=True,
                        )"""
assert old in s
s = s.replace(old, new, 1)

old2 = """    _dump_ok = bool(_dump_dir) and int(x.shape[0]) > 1"""
new2 = """    _dump_ok = bool(_dump_dir) and (
        int(x.shape[0]) > 1 or os.environ.get("ATOM_REFERENCE_DUMP_DECODE") == "1"
    )"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
ast.parse(s)
print("4-D branch logging and unconditional decode dump added")
