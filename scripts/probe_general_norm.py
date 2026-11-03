import json
from pathlib import Path
import torch
from safetensors import safe_open
from atom.model_ops.mellum_hf_norm import rmsnorm_hf_prefill

torch.set_num_threads(4)
root=Path('/workspace/logs/step0_causal_boundary/A-0003')
model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
def load(p):return torch.load(p,map_location='cpu',weights_only=True).cuda()
def weight(layer,label):
 name=f'model.layers.{layer}.{label}.weight'
 with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:
  return f.get_tensor(name).cuda()
def check(x,w,r,tag):
 summed=x if r is None else x+r
 xf=summed.float()
 ref=w*(xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+1e-6)).to(x.dtype)
 result=rmsnorm_hf_prefill(x,w,1e-6,r)
 out=result if r is None else result[0]
 d=(out.float()-ref.float()).abs()
 report={'tag':tag,'shape':list(x.shape),'different':int((d!=0).sum()),'max':float(d.max()),'mean':float(d.mean())}
 if r is not None:
  report['residual_different']=int((result[1]!=summed).sum())
 print(json.dumps(report),flush=True)
 return report
reports=[]
for i in range(28):
 folder=root/f'capture_isolated_{i}'
 for label,name in [('input_norm','input_layernorm'),('post_norm','post_attention_layernorm')]:
  x=load(folder/f'{label}_in0.pt')
  r=load(folder/f'{label}_in1.pt') if (folder/f'{label}_in1.pt').exists() else None
  reports.append(check(x,weight(i,name),r,f'L{i}.{label}'))
x=load(root/'capture_isolated_0/input_norm_in0.pt')
w=weight(0,'input_layernorm')
for rows in [1,2,8,64,512,990,1574,2240,4096,7881,12580,16000]:
 expanded=x.repeat(((rows+x.shape[0]-1)//x.shape[0],1))[:rows].contiguous()
 reports.append(check(expanded,w,None,f'rows{rows}'))
out=root.parent/'general_norm_probe.json'
out.write_text(json.dumps(reports,indent=2))
print('GENERAL_NORM_DONE',sum(r['different'] for r in reports),'differences',flush=True)
