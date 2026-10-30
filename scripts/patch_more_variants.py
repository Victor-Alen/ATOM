#!/usr/bin/env python3
"""Add two more selectable MoE variants to the isolated worktree.

ATOM_MOE_SEPARATE=1        : aiter, one fused_moe call per top-k slot, fp32 fixed-order sum
ATOM_TRITON_ALPHA=<float>  : explicit swiglu alpha for the Triton path
"""
import ast
from pathlib import Path

F = Path("/workspace/ATOM-tritonmoe/atom/model_ops/moe.py")
src = F.read_text()
NL = chr(10)

if "ATOM_MOE_SEPARATE" in src:
    print("already patched")
    raise SystemExit

# 1) separate-experts branch, before the triton branch
anchor = "        if __import__('os').environ.get('ATOM_TRITON_MOE_EXPERTS') == '1':" + NL
sep = (
    "        if __import__('os').environ.get('ATOM_MOE_SEPARATE') == '1':" + NL
    + "            import torch as _t" + NL
    + "            _terms = []" + NL
    + "            for _slot in range(int(topk_weights.shape[1])):" + NL
    + "                _terms.append(" + NL
    + "                    fused_moe(" + NL
    + "                        hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight," + NL
    + "                        topk_weight=topk_weights[:, _slot:_slot + 1].contiguous()," + NL
    + "                        topk_ids=topk_ids[:, _slot:_slot + 1].contiguous()," + NL
    + "                        expert_mask=expert_map, activation=activation," + NL
    + "                    )" + NL
    + "                )" + NL
    + "            return _t.stack([_term.float() for _term in _terms], dim=0).sum(dim=0).to(x.dtype)" + NL
    + NL
)
assert anchor in src
src = src.replace(anchor, sep + anchor, 1)

# 2) explicit alpha in the triton helper
old = (
    "    _out = _torch.empty_like(x)" + NL
    + "    return triton_kernel_fused_experts(" + NL
)
new = (
    "    _out = _torch.empty_like(x)" + NL
    + "    _alpha_env = __import__('os').environ.get('ATOM_TRITON_ALPHA')" + NL
    + "    _alpha = float(_alpha_env) if _alpha_env else 1.702" + NL
    + "    return triton_kernel_fused_experts(" + NL
)
assert old in src
src = src.replace(old, new, 1)
old2 = "        swiglu_limit=1e30," + NL
new2 = "        swiglu_limit=1e30," + NL + "        swiglu_alpha=_alpha," + NL
assert old2 in src
src = src.replace(old2, new2, 1)

F.write_text(src)
ast.parse(src)
print("variants added")
