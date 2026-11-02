from pathlib import Path
import ast

p = Path("/workspace/ATOM-tritonmoe/atom/model_ops/attention_mha.py")
s = p.read_text()

old = '''        am = fwd_ctx.attn_metadata
        ctx_lens = am.context_lens
        bt = am.block_tables'''
new = '''        am = fwd_ctx.attn_metadata
        ctx_lens = am.context_lens
        bt = am.block_tables
        if bt is None or ctx_lens is None or k_cache is None or v_cache is None:
            # Prefill metadata has no block table; keep the fused kernel there.
            if fwd_ctx.context.is_prefill:
                if self.use_flash_layout:
                    return self.prefill_attention_triton(
                        q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
                    )
                return self.prefill_attention(
                    q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
                )
            return self.paged_attention_triton(
                q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
            )'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("prefill fallback added")
