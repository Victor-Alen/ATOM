import json
from pathlib import Path
import torch
from atom.model_ops.mellum_hf_norm import rmsnorm_hf_prefill

source=Path('/workspace/logs/step0_continuation/first_divergence/atom/step_27.pt')
data=torch.load(source,map_location='cpu',weights_only=True)['tensors']
reports=[]
for key in sorted(data):
 if not key.endswith('.weight') or 'norm' not in key:
  continue
 base=key[:-7]
 if base+'.in0' not in data:
  continue
 x=data[base+'.in0'].cuda();w=data[key].cuda()
 r=data.get(base+'.in1');r=None if r is None else r.cuda()
 summed=x if r is None else x+r
 f=summed.float();ref=w*(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
 actual=rmsnorm_hf_prefill(x,w,1e-6,r,return_stats=True)
 y=actual[0]
 delta=(y.float()-ref.float()).abs()
 row={'tag':base,'different':int((delta!=0).sum()),'max':float(delta.max()),'total':delta.numel()}
 variance=f.square().mean(-1)+1e-6
 row['variance_different']=int((actual[2][:,1]!=variance).sum())
 row['rsqrt_different']=int((actual[2][:,2]!=torch.rsqrt(variance)).sum())
 if r is not None:row['residual_different']=int((actual[1]!=summed).sum())
 reports.append(row)
 print(json.dumps(row),flush=True)
assert len(reports)==57,len(reports)
Path('/workspace/logs/step0_causal_boundary/general_norm_decode.json').write_text(json.dumps(reports,indent=2))
print('DECODE_NORM_DONE',sum(r['different'] for r in reports),'differences',flush=True)
