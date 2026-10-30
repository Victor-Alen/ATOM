#!/usr/bin/env python3
"""Probe v6: test whether the Triton path drops the dominant top-k slot."""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

# helper that skips the largest-weight slot
helper_anchor = "def _atom_moe_triple("
helper = '''def _atom_moe_torch_ref_skip_top(layer, x, topk_weights, topk_ids):
    """fp32 reference that drops each row's dominant slot, to test slot loss."""
    import torch as _torch
    import torch.nn.functional as _F

    _w13 = layer._triton_w13.transpose(-1, -2).float()
    _w2 = layer._triton_w2.transpose(-1, -2).float()
    _n, _slots = topk_ids.shape
    _acc = _torch.zeros(x.shape[0], x.shape[1], dtype=_torch.float32, device=x.device)
    _xf = x.float()
    for _t in range(_n):
        _skip = int(_torch.argmax(topk_weights[_t]))
        _row = _torch.zeros(x.shape[1], dtype=_torch.float32, device=x.device)
        for _s in range(_slots):
            if _s == _skip:
                continue
            _e = int(topk_ids[_t, _s])
            _g = _F.linear(_xf[_t:_t + 1], _w13[_e])
            _gate, _up = _g.chunk(2, dim=-1)
            _inter = _F.silu(_gate) * _up
            _o = _F.linear(_inter, _w2[_e])
            _row = _row + _o[0] * float(topk_weights[_t, _s])
        _acc[_t] = _row
    return _acc.to(x.dtype)


'''
if "_atom_moe_torch_ref_skip_top" not in src:
    idx = src.index(helper_anchor)
    src = src[:idx] + helper + src[idx:]

old = "        'x_absmax': float(x.float().abs().max())," + NL
new = (
    "        'x_absmax': float(x.float().abs().max())," + NL
    + "        'ref_skip_top_absmax': float(_skipref.float().abs().max())," + NL
    + "        'd_triton_vs_ref_skip_top': _mad(_trit[0], _skipref)," + NL
    + "        'd_aiter_vs_ref_skip_top': _mad(_aiter[0], _skipref)," + NL
)
assert old in src
src = src.replace(old, new, 1)

old2 = "    _ref = [" + NL
new2 = (
    "    _skipref = _atom_moe_torch_ref_skip_top(layer, x, topk_weights, topk_ids)" + NL
    + "    _ref = [" + NL
)
assert old2 in src
src = src.replace(old2, new2, 1)

F.write_text(src)
ast.parse(src)
print("probe v6 patched")
