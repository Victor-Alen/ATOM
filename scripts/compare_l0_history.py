import json
from pathlib import Path
import torch
root=Path('/workspace/logs/step0_causal_boundary/current_A0002_step3')
a=torch.load(root/'atom/step_3.pt',map_location='cpu',weights_only=True)['tensors']
h=torch.load(root/'hf/step_3.pt',map_location='cpu',weights_only=True)['tensors']
def metric(x,y):
 x=x.reshape(-1).float();y=y.reshape(-1).float();assert x.shape==y.shape,(x.shape,y.shape)
 d=(x-y).abs();return {'different':int((d!=0).sum()),'total':d.numel(),'max':float(d.max()),'mean':float(d.mean())}
rows=[]
for step in range(4):
 prefix=f'L0.step{step}.'
 gu=a[prefix+'qkv_proj.out0'];q,k,v=gu.split([4096,512,512],-1)
 row={'step':step,'input_norm':metric(a[prefix+'qkv_proj.in0'],h[prefix+'q_proj.in0']),
      'q_proj':metric(q,h[prefix+'q_proj.out0']),'k_proj':metric(k,h[prefix+'k_proj.out0']),'v_proj':metric(v,h[prefix+'v_proj.out0']),
      'q_norm':metric(a[prefix+'q_norm.out0'],h[prefix+'q_norm.out0']),
      'k_norm':metric(a[prefix+'k_norm.out0'],h[prefix+'k_norm.out0']),
      'attention_core':metric(a[prefix+'o_proj.in0'],h[prefix+'o_proj.in0']),
      'attention_output':metric(a[prefix+'o_proj.out0'],h[prefix+'o_proj.out0'])}
 rows.append(row);print(json.dumps(row),flush=True)
(root/'l0_history_diff.json').write_text(json.dumps(rows,indent=2))
