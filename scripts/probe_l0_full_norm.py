import torch,json
from pathlib import Path
r=Path('/workspace/logs/step0_continuation/prefill990_prod');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];rows=[]
for name in ['input_norm','post_norm']:
 stem='L0.full.'+name;x=a[stem+'.in0'].cuda();res=a.get(stem+'.in1');xx=x if res is None else x+res.cuda();w=a['L0.'+name+'.weight'].cuda();f=xx.float();ref=w*(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-6)).bfloat16();actual=a[stem+'.out0'].cuda();coords=(actual!=ref).nonzero();d=(actual.float()-ref.float()).abs();first=[]
 for token,feature in coords[:10].tolist():first.append({'token':token,'feature':feature,'actual':float(actual[token,feature]),'hf_same_input':float(ref[token,feature])})
 rows.append({'name':name,'different':len(coords),'total':d.numel(),'max':float(d.max()),'mean':float(d.mean()),'first10':first})
(r/'l0_norms.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
