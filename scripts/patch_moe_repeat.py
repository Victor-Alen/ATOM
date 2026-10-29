#!/usr/bin/env python3
"""MoE repeat probe v4: routing vs accumulation, exception-safe."""
import ast, subprocess
from pathlib import Path

REPO = Path("/workspace/ATOM")
F = REPO / "atom/model_ops/moe.py"
subprocess.run(["git", "checkout", "--", str(F)], cwd=REPO, check=True)
src = F.read_text()

anchor = """            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
"""
idx = src.rindex(anchor)
block = """            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        if __import__('os').environ.get('ATOM_MOE_REPEAT') and hidden_states.shape[0] <= 64:
            try:
                import hashlib as _hl
                import json as _json
                import os as _os

                def _dg(t):
                    return _hl.sha1(
                        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    ).hexdigest()[:12]

                _fe = getattr(self.quant_method, 'fused_experts', None)
                _common = dict(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    use_grouped_topk=self.use_grouped_topk,
                    top_k=self.top_k,
                    renormalize=self.renormalize,
                    topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    e_score_correction_bias=self.e_score_correction_bias,
                    num_routing_experts=self.global_num_experts,
                    num_fused_shared_experts=getattr(self, 'num_fused_shared_experts', 0),
                    fused_shared_experts_scoring_func=self.shared_expert_scoring_func,
                    routed_scaling_factor=getattr(self, 'routed_scaling_factor', 1.0),
                )
                _w0, _i0 = FusedMoE.select_experts(**_common)
                _route = []
                for _ in range(2):
                    _w, _i = FusedMoE.select_experts(**_common)
                    _route.append((_dg(_w), _dg(_i)))
                _fixed = []
                for _ in range(5):
                    if _fe is not None:
                        _o = _fe(
                            hidden_states=hidden_states,
                            w1=self.w13_weight,
                            w2=self.w2_weight,
                            topk_weights=_w0,
                            topk_ids=_i0,
                            inplace=False,
                            activation=self.activation,
                            quant_type=QuantType.No,
                            global_num_experts=self.global_num_experts,
                            expert_map=self.expert_mask,
                        )
                    else:
                        from aiter.fused_moe import fused_moe as _fm

                        _o = _fm(
                            hidden_states=hidden_states,
                            w1=self.w13_weight,
                            w2=self.w2_weight,
                            topk_weight=_w0,
                            topk_ids=_i0,
                            expert_mask=self.expert_mask,
                            activation=self.activation,
                        )
                    _fixed.append(_dg(_o))
                torch.cuda.synchronize()
                _rec = {
                    'ok': True,
                    'rows': int(hidden_states.shape[0]),
                    'path': 'fused_experts' if _fe is not None else 'aiter_fused_moe',
                    'x_digest': _dg(hidden_states),
                    'router_digest': _dg(router_logits),
                    'route_distinct': len(set(_route)),
                    'topk_ids_digest': _dg(_i0),
                    'fixed_out_distinct': len(set(_fixed)),
                    'fixed_delta': max(
                        float(((_o - _o0).abs().max())) for _o in [self.__dict__.get('_last', _o0)] if True
                    ) if False else 0.0,
                    'fixed_outs': _fixed,
                }
            except Exception as _exc:  # noqa: BLE001
                import json as _json2
                import os as _os2

                _rec = {'ok': False, 'err': f'{type(_exc).__name__}: {_exc}'}
                _os = _os2
            try:
                with open(__import__('os').environ['ATOM_MOE_REPEAT'], 'a') as _fh:
                    _fh.write(_json.dumps(_rec) + chr(10))
            except Exception:
                pass
"""
src = src[:idx] + block + src[idx + len(anchor):]
F.write_text(src)
ast.parse(src)
print("probe v4 patched")
