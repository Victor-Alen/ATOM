from pathlib import Path
root = Path('/workspace/ATOM-general-norm/atom/model_ops')
p = root / 'layernorm.py'
s = p.read_text()
old = 'envs.ATOM_MELLUM_HF_NORM_PREFILL and x.shape==(990,2304)'
new = 'envs.ATOM_MELLUM_HF_NORM_PREFILL and x.ndim==2 and x.shape[0]>1 and x.shape[1]==2304'
assert s.count(old) == 1
p.write_text(s.replace(old,new))
p = root / 'mellum_hf_norm.py'
s = p.read_text().replace('for M990,N2304', 'for arbitrary prefill rows, N2304')
changes = {
 'assert x.shape==(990,2304)': 'assert x.ndim==2 and x.shape[0]>0 and x.shape[1]==2304',
 'out=torch.empty_like(x);rout=torch.empty_like(x) if residual is not None else None;stats=torch.empty((990,3),device=x.device,dtype=torch.float32)':
 'rows=x.shape[0];out=torch.empty(x.shape,device=x.device,dtype=x.dtype);rout=torch.empty_like(out) if residual is not None else None;stats=torch.empty((rows,3),device=x.device,dtype=torch.float32)',
 '_norm[(990,)]': '_norm[(rows,)]',
}
for old,new in changes.items():
 assert s.count(old)==1, old
 s=s.replace(old,new)
p.write_text(s)
