#!/usr/bin/env python3
"""Unshuffle expert weights per layer instead of caching them for all 28 layers
(the cached copies cost ~22 GB and caused OOM at 8k tokens)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/reference_mode.py")
s = p.read_text()
old = """            if not hasattr(exp, "_ref_w13"):
                exp._ref_w13 = _unshuffle(exp.w13_weight)
                exp._ref_w2 = _unshuffle(exp.w2_weight)
            moe = _moe_grouped(exp._ref_w13, exp._ref_w2, layer.mlp.gate.weight, h2,
                               n_experts, top_k, bool(getattr(cfg, "norm_topk_prob", True)))"""
new = """            _w13 = _unshuffle(exp.w13_weight)
            _w2 = _unshuffle(exp.w2_weight)
            moe = _moe_grouped(_w13, _w2, layer.mlp.gate.weight, h2,
                               n_experts, top_k, bool(getattr(cfg, "norm_topk_prob", True)))
            del _w13, _w2"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("per-layer unshuffle (no cache)")
