from pathlib import Path
p=Path('/workspace/ATOM-general-norm/atom/model_ops/attention_mha.py')
s=p.read_text()
old='''                q, k = self.rotary_emb(position, q, k)
'''
assert s.count(old)==1
new='''                if __import__('os').environ.get('ATOM_MELLUM_EXACT_ROPE')=='1':
                    from atom.model_ops.mellum_hf_rope import rotate
                    q,k=rotate(get_current_atom_config().hf_config,self.layer_num,position,q,k)
                else:
                    q, k = self.rotary_emb(position, q, k)
'''
p.write_text(s.replace(old,new))
