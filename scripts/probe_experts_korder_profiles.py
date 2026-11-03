import torch,triton as tr,triton.language as tl,json
from pathlib import Path
from safetensors import safe_open
@tr.jit
def gateup(X,W,Y,M,N:tl.constexpr,STAGGER:tl.constexpr,PERM:tl.constexpr,TILE:tl.constexpr):
 rm=tl.program_id(0)*32+tl.arange(0,32);rn=tl.program_id(1)*32+tl.arange(0,32);rk=tl.arange(0,64)
 if STAGGER==1:start=(tl.program_id(0)*32//TILE)%16
 elif STAGGER==2:start=(tl.program_id(1)*32//192)%16
 else:start=0
 acc=tl.zeros((32,32),tl.float32)
 for chunk in range(36):
  if PERM==1:rr=(rk&~12)|((rk&4)<<1)|((rk&8)>>1)
  elif PERM==2:rr=(rk&~20)|((rk&4)<<2)|((rk&16)>>2)
  else:rr=rk
  kk=((chunk+start)%36)*64+rr
  x=tl.load(X+rm[:,None]*2304+kk[None,:],mask=rm[:,None]<M,other=0)
  w=tl.load(W+rn[None,:]*2304+kk[:,None],mask=rn[None,:]<N,other=0)
  acc=tl.dot(x,w,acc)
 tl.store(Y+rm[:,None]*N+rn[None,:],acc.to(tl.bfloat16),mask=(rm[:,None]<M)&(rn[None,:]<N))
r=Path('/workspace/logs/step0_continuation/norm990_active_prod');hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors'];iso=torch.load(r/'ck_stage_isolation.pt',weights_only=False);h=hf['L0.full.post_norm.out0'].reshape(990,2304).cuda();model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');wm=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];rows=[]
profile=json.loads((r/'hf_grouped_gateup_profile_summary.json').read_text());active=iso['ids'].flatten().bincount(minlength=64).nonzero().flatten().tolist();km=dict(zip(active,[k['name'] for k in profile['kernels']]))
for e in [55,7,23,49]:
 mask=iso['ids']==e;tokens=mask.nonzero()[:,0].cuda();x=h[tokens].contiguous();ref=torch.cat([iso['gate_hf'][mask],iso['up_hf'][mask]],-1).cuda();weights=[]
 for proj in ['gate_proj','up_proj']:
  key=f'model.layers.0.mlp.experts.{e}.{proj}.weight'
  with safe_open(model/wm[key],framework='pt',device='cpu') as f:weights.append(f.get_tensor(key))
 w=torch.cat(weights).cuda();m=x.shape[0];direct=int((torch.nn.functional.linear(x,w)!=ref).sum());mi32='MI32x32' in km[e];tile=128 if mi32 else 192
 for perm in [0,1,2]:
  out=torch.empty_like(ref);gateup[(tr.cdiv(m,32),56)](x,w,out,m,1792,1,perm,tile,num_warps=1,matrix_instr_nonkdim=32 if mi32 else 16,enable_fp_fusion=False);row={'expert':e,'m':m,'tile':tile,'mi':32 if mi32 else 16,'perm':perm,'direct_hf_different':direct,'gate_different':int((out[:,:896]!=ref[:,:896]).sum()),'up_different':int((out[:,896:]!=ref[:,896:]).sum())};rows.append(row);print(row,flush=True)
(r/'experts_korder_profiles.json').write_text(json.dumps(rows,indent=2)+'\n')
