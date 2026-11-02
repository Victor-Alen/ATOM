from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()

old = '''        return o

    def dispatch_backend(self, fwd_ctx: ForwardContext):'''
new = '''        if __import__("os").environ.get("ATOM_EAGER_ONLY") != "1":
            ref = None
            if fwd_ctx.context.is_prefill:
                ref = (
                    self.prefill_attention_triton
                    if self.use_flash_layout
                    else self.prefill_attention
                )(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx)
            else:
                ref = self.paged_attention_triton(
                    q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
                )
            if not globals().get("_eager_diff_once"):
                globals()["_eager_diff_once"] = True
                d = (o.float() - ref.float()).abs()
                print(
                    "[eager] max|eager-fused| = %.6g  mean = %.6g  shapes %s / %s"
                    % (float(d.max()), float(d.mean()), tuple(o.shape), tuple(ref.shape)),
                    flush=True,
                )
            return ref
        return o

    def dispatch_backend(self, fwd_ctx: ForwardContext):'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("diff diagnostic wired (returns fused output)")
