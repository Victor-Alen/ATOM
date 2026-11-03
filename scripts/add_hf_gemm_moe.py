from pathlib import Path
p=Path('/workspace/ATOM-general-norm/atom/model_ops/moe.py')
s=p.read_text()
anchor='        if getattr(layer, "model_type", None) == "mellum":\n'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'''            if __import__('os').environ.get('ATOM_MELLUM_HF_GEMM_MOE')=='1':
                from atom.model_ops.mellum_hf_gemm_moe import apply
                return apply(layer,x,router_logits)
''')
p.write_text(s)
