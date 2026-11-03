import json,torch
from pathlib import Path
from safetensors import safe_open
import torch.nn.functional as F
root=Path('/workspace/logs/step0_continuation/norm_hf_divergence')
a=torch.load(root/'atom/step_3.pt',map_location='cpu',weights_only=False)['tensors']
m=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');wm=json.loads((m/'model.safetensors.index.json').read_text())['weight_map']
def w(name):
 name='model.layers.0.self_attn.'+name
 with safe_open(m/wm[name],framework='pt',device='cpu') as f:return f.get_tensor(name).cuda()
def metric(x,y):
 d=(x.float()-y.to(x.device).float()).abs();return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
q,k,v,o=[w(n+'_proj.weight') for n in ['q','k','v','o']];qn,kn=w('q_norm.weight'),w('k_norm.weight')
rows=[]
with torch.inference_mode():
 for step in range(4):
  stem=f'L0.step{step}.';x=a[stem+'qkv_proj.in0'].cuda();actual=a[stem+'qkv_proj.out0'].cuda();aq,ak,av=actual.split([4096,512,512],-1)
  row={'step':step,'tokens':x.shape[0],'q_proj':metric(F.linear(x,q),aq),'k_proj':metric(F.linear(x,k),ak),'v_proj':metric(F.linear(x,v),av)}
  for name,weight in [('q_norm',qn),('k_norm',kn)]:
   xx=a[stem+name+'.in0'].cuda();f=xx.float();ref=weight*(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-6)).bfloat16()
   row[name]=metric(ref,a[stem+name+'.out0'])
  row['o_proj']=metric(F.linear(a[stem+'o_proj.in0'].cuda(),o),a[stem+'o_proj.out0'])
  rows.append(row)
(root/'l0_detail_comparison.json').write_text(json.dumps({'rows':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
