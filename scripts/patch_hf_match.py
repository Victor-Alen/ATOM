#!/usr/bin/env python3
"""Env-gated switches that reproduce HF's precision-loss points in ATOM.

ATOM_LOGITS_BF16_ARGMAX=1 : argmax on bf16-quantised logits, like HF's bf16 lm_head
ATOM_MOE_HF_ORDER=1       : per-expert bf16 accumulate in ascending expert id, like HF
ATOM_NORM_HF_BF16=1       : RMSNorm casts to bf16 before the weight multiply, like HF
"""
import ast
from pathlib import Path

WT = Path("/workspace/ATOM-tritonmoe")
NL = chr(10)

# ---------- (1) sampler: bf16 logits before argmax ----------
S = WT / "atom/model_ops/sampler.py"
s = S.read_text()
if "ATOM_LOGITS_BF16_ARGMAX" not in s:
    if "import os" not in s.split("class ")[0]:
        s = "import os" + NL + s
    old = (
        "        # Temperature 0 must be a true argmax: the plain-temperature path" + NL
        + "        # below runs a Gumbel race with fresh noise on every call." + NL
        + "        if all_greedy:" + NL
        + "            return logits.argmax(dim=-1).to(torch.int)" + NL
    )
    new = (
        "        # Temperature 0 must be a true argmax: the plain-temperature path" + NL
        + "        # below runs a Gumbel race with fresh noise on every call." + NL
        + "        if all_greedy:" + NL
        + "            if os.environ.get('ATOM_LOGITS_BF16_ARGMAX') == '1':" + NL
        + "                logits = logits.to(torch.bfloat16).float()" + NL
        + "            return logits.argmax(dim=-1).to(torch.int)" + NL
    )
    assert old in s, "sampler anchor missing"
    s = s.replace(old, new, 1)
    S.write_text(s)
    ast.parse(s)
    print("sampler: bf16 argmax switch added")

# ---------- (2) MoE: HF-equivalent per-expert loop ----------
F = WT / "atom/model_ops/moe.py"
m = F.read_text()
if "ATOM_MOE_HF_ORDER" not in m:
    anchor = "        if __import__('os').environ.get('ATOM_MOE_SEPARATE') == '1':" + NL
    branch = (
        "        if __import__('os').environ.get('ATOM_MOE_HF_ORDER') == '1':" + NL
        + "            return _atom_moe_hf_order(layer, x, topk_weights, topk_ids)" + NL
        + NL
    )
    assert anchor in m, "moe anchor missing"
    m = m.replace(anchor, branch + anchor, 1)
    helper = NL + NL + '''def _atom_moe_hf_order(layer, x, topk_weights, topk_ids):
    """Bit-for-bit imitation of HF Mellum's sparse MoE combine.

    HF keeps the accumulator in bf16 and adds each expert's contribution in
    ascending expert id via index_add_. aiter ships shuffled weights, so the
    (16,16) shuffle is inverted and cached on the layer once.
    """
    import torch as _torch

    if not getattr(layer, '_hf_moe_ready', False):
        with _torch.no_grad():
            def _unshuffle(w):
                E, n, k = w.shape
                return (
                    w.view(E, n // 16, k // 32, 4, 16, 8)
                    .permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .view(E, n, k)
                )

            layer._hf_w13 = _unshuffle(layer.w13_weight)  # [E, 2I, H]
            layer._hf_w2 = _unshuffle(layer.w2_weight)    # [E, H, I]
            layer._hf_moe_ready = True

    _final = _torch.zeros_like(x)
    _n_experts = layer._hf_w13.shape[0]
    for _e in range(_n_experts):
        _mask = topk_ids == _e
        if not bool(_mask.any()):
            continue
        _tok_idx, _slot_idx = _torch.where(_mask)
        _cur = x[_tok_idx]
        _gate_up = _cur @ layer._hf_w13[_e].transpose(-1, -2)
        _gate, _up = _gate_up.chunk(2, dim=-1)
        _inter = _torch.nn.functional.silu(_gate) * _up
        _out = _inter @ layer._hf_w2[_e].transpose(-1, -2)
        _out = _out * topk_weights[_tok_idx, _slot_idx, None]
        _final.index_add_(0, _tok_idx, _out.to(_final.dtype))
    return _final
'''
    m = m + helper
    F.write_text(m)
    ast.parse(m)
    print("moe: HF-order bf16 switch added")

# ---------- (3) RMSNorm: bf16 weight multiply ----------
L = WT / "atom/model_ops/layernorm.py"
l = L.read_text()
if "ATOM_NORM_HF_BF16" not in l:
    anchor = "        if self.x_pad_to_multiple > 0:" + NL
    branch = (
        "        if __import__('os').environ.get('ATOM_NORM_HF_BF16') == '1':" + NL
        + "            _x = x if residual is None else x + residual" + NL
        + "            _f = _x.float()" + NL
        + "            _var = _f.pow(2).mean(-1, keepdim=True)" + NL
        + "            _norm = (_f * _torch.rsqrt(_var + self.eps)).to(_x.dtype)" + NL
        + "            _out = self.weight * _norm" + NL
        + "            if residual is None:" + NL
        + "                return _out" + NL
        + "            return _out, _x" + NL
        + NL
    )
    assert anchor in l, "norm anchor missing"
    l = l.replace(anchor, branch + anchor, 1)
    if "import torch as _torch" not in l:
        l = "import torch as _torch" + NL + l
    L.write_text(l)
    ast.parse(l)
    print("norm: HF bf16 switch added")
