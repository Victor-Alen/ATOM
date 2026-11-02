from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()
old = '''        attn_impl = self.dispatch_backend(fwd_ctx)

        o = attn_impl(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx)'''
new = '''        _dp = __import__("os").environ.get("ATOM_ROPE_DUMP")
        if _dp and self.layer_num == 0 and not globals().get("_rope_dumped"):
            globals()["_rope_dumped"] = True
            _t = __import__("torch")
            _t.save(
                {
                    "pos": position.detach().cpu(),
                    "q_post": q.detach().float().cpu(),
                    "k_post": k.detach().float().cpu(),
                    "v_post": v.detach().float().cpu(),
                },
                _dp,
            )
            print("[rope-dump] saved", _dp, "q", tuple(q.shape), flush=True)

        attn_impl = self.dispatch_backend(fwd_ctx)

        o = attn_impl(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx)'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("rope dump added")
