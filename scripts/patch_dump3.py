#!/usr/bin/env python3
"""Dump layer-0 inputs (embedding, post-norm, attention out, mlp out) for the first call."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
if "ATOM_REFERENCE_DUMP2" not in s:
    old = """            h = _hf_norm(x[lo:hi], layer.input_layernorm.weight, eps)"""
    new = """            h = _hf_norm(x[lo:hi], layer.input_layernorm.weight, eps)
            if li == 0 and os.environ.get("ATOM_REFERENCE_DUMP2") == "1" and "d2" not in _ROT:
                import torch as _t2

                _t2.save(
                    {
                        "ids": input_ids.detach().cpu(),
                        "x_in": x.detach().cpu(),
                        "h_ln": h.detach().cpu(),
                        "ln_w": layer.input_layernorm.weight.detach().cpu(),
                    },
                    "/workspace/logs/atom_l0_in.pt",
                )
                _ROT["d2"] = True
                print("[ref] dumped layer0 inputs", flush=True)"""
    assert old in s
    s = s.replace(old, new, 1)
    p.write_text(s)
    ast.parse(s)
    print("layer0 input dump added")
