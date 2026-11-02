from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()

# wrap the body so any exception logs once and falls back to the fused kernel
old = '''        o = _t.empty_like(q)'''
new = '''        if not globals().get("_eager_dbg"):
            globals()["_eager_dbg"] = True
            print("[eager] k_cache", tuple(k_cache.shape), "v_cache", tuple(v_cache.shape),
                  "bt", None if bt is None else tuple(bt.shape),
                  "ctx", tuple(ctx_lens.shape), "cu", tuple(cu.shape),
                  "q", tuple(q.shape), "scale", type(self.scale).__name__,
                  getattr(self.scale, "shape", self.scale), flush=True)

        o = _t.empty_like(q)'''
assert old in s
s = s.replace(old, new, 1)

# also guard scale usage
s = s.replace('sc = _t.einsum("qhd,khd->hqk", qb * self.scale, kb)',
              'sc = _t.einsum("qhd,khd->hqk", qb * float(self.scale), kb)')
p.write_text(s)
ast.parse(s)
print("debug print + scale coercion added")
