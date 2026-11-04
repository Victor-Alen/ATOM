import json
import sys
from pathlib import Path
import torch
from aiter.rotary_embedding import get_rope
from atom.model_ops.mellum_hf_rope import rotate
sys.path.insert(0,'/workspace/ATOM-general-norm/scripts')
from hf_reference_generate import MODEL_PATH,load_mellum_classes
Config,_=load_mellum_classes()
hf=sys.modules['transformers.models.mellum.modeling_mellum']
cfg=Config.from_pretrained(MODEL_PATH)
reference=hf.MellumRotaryEmbedding(cfg).cuda()
torch.manual_seed(19)
results=[]
def metric(a,b):
 d=(a.float()-b.float()).abs()
 return {'different':int((d!=0).sum()),'max':float(d.max()),'total':d.numel()}
for layer_type in sorted(set(cfg.layer_types)):
 base=cfg.rope_parameters[layer_type]['rope_theta']
 with torch.device('cuda'):
  operator=get_rope(128,rotary_dim=128,max_position=cfg.max_position_embeddings,base=base,rope_scaling=dict(cfg.rope_parameters[layer_type]),dtype=torch.bfloat16)
 for n,start in [(990,0),(1574,0),(2240,1000),(12580,0)]:
  positions=torch.arange(start,start+n,device='cuda')
  q=torch.randn(n,32,128,device='cuda',dtype=torch.bfloat16)
  k=torch.randn(n,4,128,device='cuda',dtype=torch.bfloat16)
  cos,sin=reference(q,positions[None],layer_type)
  rq,rk=hf.apply_rotary_pos_emb(q.transpose(0,1)[None],k.transpose(0,1)[None],cos,sin)
  rq=rq[0].transpose(0,1);rk=rk[0].transpose(0,1)
  hq,hk=operator.forward_hip(positions,q.clone(),k.clone())
  nq,nk=operator.forward_native(positions,q,k)
  eq,ek=rotate(cfg,cfg.layer_types.index(layer_type),positions,q,k)
  row={'type':layer_type,'n':n,'start':start,
       'cos':metric(operator.cos_cache[positions].reshape(n,64),cos[0,:,:64]),
       'hip_q':metric(hq,rq),'hip_k':metric(hk,rk),
       'native_q':metric(nq,rq),'native_k':metric(nk,rk),
       'exact_q':metric(eq,rq),'exact_k':metric(ek,rk)}
  results.append(row);print(json.dumps(row),flush=True)
Path('/workspace/logs/step0_causal_boundary/rope_backends_probe.json').write_text(json.dumps(results,indent=2))
print('ROPE_BACKENDS_DONE',flush=True)
assert all(r['exact_q']['different']==r['exact_k']['different']==0 for r in results)
