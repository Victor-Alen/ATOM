#!/usr/bin/env python3
"""Triple MoE probe v2: layers 0-3, with routing diagnostics, returns the aiter result
so the rest of the forward stays comparable."""
import ast, re
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

# widen the guard
old_guard = (
    "        if (__import__('os').environ.get('ATOM_MOE_TRIPLE')" + NL
    + "                and '.layers.0.' in str(getattr(layer, 'layer_name', ''))" + NL
    + "                and x.shape[0] <= 64):" + NL
)
if old_guard in src:
    new_guard = (
        "        if (__import__('os').environ.get('ATOM_MOE_TRIPLE')" + NL
        + "                and re.search(r'\\.layers\\.[0-3]\\.', str(getattr(layer, 'layer_name', '')))" + NL
        + "                and x.shape[0] <= 64):" + NL
    )
    src = src.replace(old_guard, new_guard, 1)
    if "import re" not in src.split("class ")[0]:
        src = "import re" + NL + src
    print("guard widened to layers 0-3")
else:
    print("guard already widened or missing")

# add diagnostics + return aiter result
old_ret = "    return _trit[0]" + NL
new_ret = (
    "    _rec['hist'] = [int(v) for v in _routing_hist]" + NL
    + "    _rec['zero_experts'] = int((_routing_hist == 0).sum())" + NL
    + "    _rec['n_expts_tot'] = int(_n_tot)" + NL
    + "    _rec['expert_map_none'] = expert_map is None" + NL
    + "    with open(path, 'a') as _fh2:" + NL
    + "        _fh2.write(_json.dumps(_rec) + chr(10))" + NL
    + "    return _aiter[0]" + NL
)
if old_ret in src:
    src = src.replace(old_ret, new_ret, 1)
    print("returns aiter result + logs hist")

# capture routing/hist inside _atom_moe_triple
old_route = (
    "    _aiter = [" + NL
)
new_route = (
    "    from atom.model_ops.fused_moe_triton import routing_from_topk as _rft" + NL
    + "    _n_tot = layer.w13_weight.shape[0]" + NL
    + "    _routing_dbg, _g_dbg, _s_dbg = _rft(topk_weights, topk_ids, _n_tot)" + NL
    + "    _routing_hist = _routing_dbg.hist" + NL
    + "    _rec_dbg = {" + NL
    + "        'hist_head': [int(v) for v in _routing_hist[:16]]," + NL
    + "    }" + NL
    + "    _aiter = [" + NL
)
if old_route in src:
    src = src.replace(old_route, new_route, 1)
    print("routing diagnostics captured")

# merge debug dict into _rec
old_rec = "    _rec = {" + NL
new_rec = "    _rec = dict(_rec_dbg)" + NL + "    _rec.update({" + NL
if old_rec in src:
    src = src.replace(old_rec, new_rec, 1)
    # close the dict: the original block ends with "    }" right before the two new lines
    src = src.replace(
        "        'triton_spread': max(_mad(_trit[0], t) for t in _trit)," + NL + "    }" + NL,
        "        'triton_spread': max(_mad(_trit[0], t) for t in _trit)," + NL + "    })" + NL,
        1,
    )
    print("record merged")

# remove the earlier single write (it now duplicates)
dup = (
    "    _torch.cuda.synchronize()" + NL
    + "    with open(path, 'a') as _fh:" + NL
    + "        _fh.write(_json.dumps(_rec) + chr(10))" + NL
)
if dup in src:
    src = src.replace(dup, "    _torch.cuda.synchronize()" + NL, 1)
    print("dedup write")

F.write_text(src)
ast.parse(src)
print("probe v2 ok")
