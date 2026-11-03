from pathlib import Path
p=Path('/workspace/ATOM-general-norm/atom/model_ops/attention_mha.py')
s=p.read_text()
anchor='    def dispatch_backend(self, fwd_ctx: ForwardContext):\n'
method='''    def attention_bf16_hf_order(self,q,k,v,k_cache,v_cache,k_scale,v_scale,fwd_ctx):
        from atom.model_ops.mellum_bf16_attention import attention
        return attention(q,k,v,k_cache,v_cache,fwd_ctx.attn_metadata,self.scale,self.sliding_window)

'''
assert s.count(anchor)==1
s=s.replace(anchor,method+anchor)
anchor='        ctx = fwd_ctx.context\n'
# Limit replacement to dispatch_backend, not other context consumers.
start=s.index('    def dispatch_backend(')
head,tail=s[:start],s[start:]
assert tail.count(anchor)>=1
tail=tail.replace(anchor,anchor+'''
        if (__import__('os').environ.get('ATOM_MELLUM_BF16_HF_ATTENTION')=='1'
                and self.head_dim==128 and self.kv_cache_dtype=='bf16'
                and not self.use_flash_layout and self.sinks is None
                and getattr(get_current_atom_config().hf_config,'model_type',None)=='mellum'):
            return self.attention_bf16_hf_order
''',1)
p.write_text(head+tail)
