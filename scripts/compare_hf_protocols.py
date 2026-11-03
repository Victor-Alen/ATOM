import torch,json
from pathlib import Path
root=Path('/workspace/logs/step0_continuation/official_generate_first29')
a=torch.load('/workspace/logs/step0_continuation/hf_attn_first29/hf/step_29.pt',weights_only=False)['tensors'];b=torch.load(root/'hf/step_29.pt',weights_only=False)['tensors']
rows=[]
for layer in range(28):
 for tag in ['input_norm.in0','input_norm.out0','attn.out0','post_norm.out0','mlp.out0','layer.out0']:
  name=f'L{layer}.{tag}';x=a[name].float();y=b[name].float();d=(x-y).abs();rows.append({'name':name,'different':int((d!=0).sum()),'max':float(d.max()),'mean':float(d.mean())})
r={'manual_positions':a.get('model.positions').tolist(),'generate_positions':b.get('model.positions').tolist(),'first_changed':next((r for r in rows if r['different']),None),'rows':rows,'official_forward_kwargs':b.get('model.forward_kwargs')}
(root/'manual_vs_generate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
