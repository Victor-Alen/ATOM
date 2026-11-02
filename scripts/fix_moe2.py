from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = '''                _tk = int(_exp.top_k)
                _ne = int(_exp.num_experts)
                _norm = bool(getattr(_exp, "renormalize", True))'''
new = '''                _tk = int(getattr(_exp, "top_k", getattr(_exp, "topk", 8)))
                _ne = int(_exp.w13_weight.shape[0])
                _norm = bool(getattr(_exp, "renormalize", True))
                if "prod-moe-attrs" not in globals():
                    globals()["prod-moe-attrs"] = True
                    print(
                        "[prod-moe-ref] E=%d top_k=%d norm=%s attrs=%s"
                        % (
                            _ne,
                            _tk,
                            _norm,
                            [a for a in dir(_exp) if any(k in a.lower() for k in ("top", "norm", "expert", "w13", "w2"))],
                        ),
                        flush=True,
                    )'''
assert old in s
p.write_text(s.replace(old, new, 1))
ast.parse(p.read_text())
print("patched with robust attribute access")
