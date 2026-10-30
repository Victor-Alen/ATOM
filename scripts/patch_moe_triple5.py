#!/usr/bin/env python3
"""Probe v5: characterise the layer-2 mismatch - finiteness, magnitudes, and the
token/position of the largest reference element."""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

old = "        'd_direct_vs_ref': _mad(_direct[0], _ref[0])," + NL
new = (
    "        'd_direct_vs_ref': _mad(_direct[0], _ref[0])," + NL
    + "        'absmax_aiter': float(_aiter[0].float().abs().max())," + NL
    + "        'absmax_triton': float(_trit[0].float().abs().max())," + NL
    + "        'absmax_ref': float(_ref[0].float().abs().max())," + NL
    + "        'nonfinite_aiter': int((~_torch.isfinite(_aiter[0].float())).sum())," + NL
    + "        'nonfinite_triton': int((~_torch.isfinite(_trit[0].float())).sum())," + NL
    + "        'nonfinite_ref': int((~_torch.isfinite(_ref[0].float())).sum())," + NL
    + "        'ref_argmax_flat': int(_ref[0].float().abs().reshape(-1).argmax())," + NL
    + "        'x_absmax': float(x.float().abs().max())," + NL
    + "        'topk_ids_per_row': [[int(v) for v in row] for row in topk_ids.tolist()]," + NL
    + "        'topk_w_per_row': [[round(float(v), 5) for v in row] for row in topk_weights.tolist()]," + NL
)
assert old in src
src = src.replace(old, new, 1)
F.write_text(src)
ast.parse(src)
print("probe v5 patched")
