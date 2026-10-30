#!/usr/bin/env python3
"""Fix the triple probe diagnostics: RoutingData has no .hist; use bincount."""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

old = (
    "    from atom.model_ops.fused_moe_triton import routing_from_topk as _rft" + NL
    + "    _n_tot = layer.w13_weight.shape[0]" + NL
    + "    _routing_dbg, _g_dbg, _s_dbg = _rft(topk_weights, topk_ids, _n_tot)" + NL
    + "    _routing_hist = _routing_dbg.hist" + NL
    + "    _rec_dbg = {" + NL
    + "        'hist_head': [int(v) for v in _routing_hist[:16]]," + NL
    + "    }" + NL
)
new = (
    "    _n_tot = layer.w13_weight.shape[0]" + NL
    + "    _routing_hist = torch.bincount(" + NL
    + "        topk_ids.reshape(-1).long(), minlength=_n_tot" + NL
    + "    )" + NL
    + "    _rec_dbg = {" + NL
    + "        'hist_head': [int(v) for v in _routing_hist[:16]]," + NL
    + "    }" + NL
)
if old in src:
    src = src.replace(old, new, 1)
    print("hist via bincount")
else:
    print("pattern not found -- check file")
    import sys; sys.exit(1)

F.write_text(src)
ast.parse(src)
print("ok")
