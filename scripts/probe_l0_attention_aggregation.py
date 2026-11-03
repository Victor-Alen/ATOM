import json,sys,torch,os
from pathlib import Path
from types import SimpleNamespace
from hf_reference_generate import load_mellum_classes,MODEL_PATH
Config,_=load_mellum_classes();m=sys.modules['transformers.models.mellum.modeling_mellum'];cfg=Config.from_pretrained(MODEL_PATH)
root=Path(os.environ.get('MELLUM_TRACE_ROOT','/workspace/logs/step0_continuation/norm_hf_divergence'));target=int(os.environ.get('MELLUM_TRACE_STEP','3'));a=torch.load(root/'atom'/f'step_{target}.pt',map_location='cpu',weights_only=False)['tensors']
def metric(x,y):
 y=y.to(x.device).reshape(x.shape);d=(x.float()-y.float()).abs();return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
def heads(x,h):return x.reshape(-1,h,128).transpose(0,1).unsqueeze(0)
rope=m.MellumRotaryEmbedding(cfg,device='cuda');obj=SimpleNamespace(num_key_value_groups=8,training=False)
rows=[];pk=[];pv=[];hk=[]
with torch.inference_mode():
 for step in range(target+1):
  stem=f'L0.step{step}.';q=a[stem+'attn.inquery'].cuda();k=a[stem+'attn.inkey'].cuda();pos=a[stem+'attn.inpositions'].cuda().reshape(1,-1)
  cos,sin=rope(q,pos,cfg.layer_types[0]);hq,hkk=m.apply_rotary_pos_emb(heads(q,32),heads(k,4),cos,sin)
  pq=heads(a[stem+'attn.afterquery'].cuda(),32);kk=heads(a[stem+'attn.afterkey'].cuda(),4)
  vv=heads(a[stem+'qkv_proj.out0'].cuda()[...,4608:],4)
  pk.append(kk);pv.append(vv);hk.append(hkk)
  # Last query has no future keys; all 993 keys fit in the L0 window.
  val=torch.cat(pv,dim=2)
  fixed,_=m.eager_attention_forward(obj,pq[:,:,-1:],torch.cat(pk,dim=2),val,None,128**-0.5)
  hfrope,_=m.eager_attention_forward(obj,hq[:,:,-1:],torch.cat(hk,dim=2),val,None,128**-0.5)
  actual=a[stem+'o_proj.in0'][-1:].cuda()
  rows.append({'step':step,'q_rope':metric(hq,pq),'k_rope':metric(hkk,kk),
    'eager_on_production_rotated_qkv_vs_actual_attention':metric(fixed.reshape(1,-1),actual),
    'eager_with_hf_rope_on_production_normalized_qkv_vs_actual_attention':metric(hfrope.reshape(1,-1),actual)})
(root/'l0_attention_aggregation.json').write_text(json.dumps({'rows':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))

