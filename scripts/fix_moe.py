from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
start = s.index('        router_logits = self.gate(hidden_states)')
end = s.index('        if self.tp_size > 1 and not ENABLE_ALLREDUCE_RMSNORM_FUSION:', start)
new = '''        router_logits = self.gate(hidden_states)
        _prod_moe = __import__("os").environ.get("ATOM_PROD_MOE")
        if _prod_moe == "ref":
            try:
                from atom.models.reference_mode import _moe_grouped, _unshuffle

                _exp = self.experts
                _w13 = _unshuffle(_exp.w13_weight)
                _w2 = _unshuffle(_exp.w2_weight)
                _tk = int(_exp.top_k)
                _ne = int(_exp.num_experts)
                _norm = bool(getattr(_exp, "renormalize", True))
                final_hidden_states = _moe_grouped(
                    _w13, _w2, self.gate.weight, hidden_states, _ne, _tk, _norm
                )
            except Exception as _e:  # noqa: BLE001
                import traceback as _tb

                print("[prod-moe-ref] failed, falling back:", repr(_e), flush=True)
                _tb.print_exc()
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, router_logits=router_logits
                )
        else:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
'''
s = s[:start] + new + s[end:]
p.write_text(s)
ast.parse(s)
print("clean rewrite done")
