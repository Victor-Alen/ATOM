import torch,triton as tr,triton.language as tl,json
from triton.language.extra.hip import libdevice
from pathlib import Path
@tr.jit
def probe(X,E,P,D,MODE:tl.constexpr):
 lane=tl.arange(0,64);mx=tl.full((64,),float('-inf'),tl.float32)
 for it in tl.static_range(16):mx=tl.maximum(mx,tl.load(X+it*64+lane))
 mx=tl.max(mx,0);values=();acc=tl.full((64,),0,tl.float32)
 for it in tl.static_range(16):
  x=tl.load(X+it*64+lane)-mx
  if MODE==0:e=tl.exp(x)
  elif MODE==1 or MODE==3:e=libdevice.exp(x)
  else:e=libdevice.fast_expf(x)
  values+=(e,);acc=acc+e;tl.store(E+it*64+lane,e)
 if MODE==3:
  for level in tl.static_range(6):
   acc=acc+tl.gather(acc,lane^(32>>level),0)
  den=tl.sum(tl.where(lane==0,acc,0),0)
 else:den=tl.sum(acc,0)
 tl.store(D,den)
 for it in tl.static_range(16):tl.store(P+it*64+lane,tl.div_rn(values[it],den))
r=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');c=torch.load(r/'prefill_candidate.pt',weights_only=False);score=c['ref_scores'][21,36].cuda().float();score[37:]=float('-inf');x=torch.full((1024,),float('-inf'),device='cuda');x[:990]=score
ref=x.softmax(-1);ref990=score.softmax(-1);e=(x-x.max()).exp();rows=[]
for mode in range(4):
 ee=torch.empty_like(x);pp=torch.empty_like(x);d=torch.empty((),device='cuda');probe[(1,)](x,ee,pp,d,mode,num_warps=1,enable_fp_fusion=False)
 rows.append({'mode':mode,'denom':float(d),'torch_exp_sum':float(e.sum()),'exp_different':int((ee!=e).sum()),'p_different':int((pp!=ref).sum()),'ref31':float(ref[31]),'ref990_31':float(ref990[31]),'p31':float(pp[31]),'p31_bf16':float(pp[31].bfloat16()),'e31':float(ee[31]),'torch_e31':float(e[31])})
(r/'softmax_scalar.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
