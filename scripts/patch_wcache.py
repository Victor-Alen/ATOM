#!/usr/bin/env python3
"""Cache the unshuffled expert weights when memory allows (ATOM_REFERENCE_CACHE_W=1).

Per call we were allocating/freeing 528 MB + 117 MB per layer, 28 times; with plenty
of VRAM the cache removes that churn entirely.
"""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """            _w13 = _unshuffle(exp.w13_weight)
            _w2 = _unshuffle(exp.w2_weight)
            moe = _moe_grouped(_w13, _w2, layer.mlp.gate.weight, h2,
                               n_experts, top_k, bool(getattr(cfg, "norm_topk_prob", True)))
            del _w13, _w2"""
new = """            if os.environ.get("ATOM_REFERENCE_CACHE_W") == "1":
                if not hasattr(exp, "_ref_w13"):
                    exp._ref_w13 = _unshuffle(exp.w13_weight)
                    exp._ref_w2 = _unshuffle(exp.w2_weight)
                _w13, _w2 = exp._ref_w13, exp._ref_w2
            else:
                _w13 = _unshuffle(exp.w13_weight)
                _w2 = _unshuffle(exp.w2_weight)
            moe = _moe_grouped(_w13, _w2, layer.mlp.gate.weight, h2,
                               n_experts, top_k, bool(getattr(cfg, "norm_topk_prob", True)))
            if os.environ.get("ATOM_REFERENCE_CACHE_W") != "1":
                del _w13, _w2"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("expert-weight cache re-enabled behind a flag")
