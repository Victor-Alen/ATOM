#!/usr/bin/env python3
"""Wire the env-gated layer trace into MellumModel.__init__ (idempotent)."""
import ast
from pathlib import Path

F = Path("/workspace/ATOM/atom/models/mellum.py")
src = F.read_text()
NL = chr(10)

if "install_layer_trace" not in src:
    anchor = "        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory("
    idx = src.index(anchor)
    # insert right after the constructor body that sets make_empty_intermediate_tensors
    end = src.index("    def get_input_embeddings", idx)
    insert = (
        "        if __import__('os').environ.get('ATOM_LAYER_TRACE_DIR'):" + NL
        + "            from atom.models.layer_trace import install_layer_trace" + NL
        + "" + NL
        + "            install_layer_trace(self)" + NL
        + "" + NL
    )
    src = src[:end] + insert + src[end:]
    F.write_text(src)
    ast.parse(src)
    print("mellum.py wired for layer trace")
else:
    print("already wired")
