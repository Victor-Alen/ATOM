import torch,triton as tr,triton.language as tl,json
from pathlib import Path
from safetensors import safe_open
@tr.jit
def gateup(X,W,Y,M,N:tl.constexpr,PERM:tl.constexpr,TILE:tl.constexpr,DEPTH:tl.constexpr,LSU:tl.constexpr,BM:tl.constexpr):
 rm=tl.program_id(0)*BM+tl.arange(0,BM);rn=tl.program_id(1)*32+tl.arange(0,32);rk=tl.arange(0,64)
 if PERM==1:rr=(rk&~12)|((rk&4)<<1)|((rk&8)>>1)
 else:rr=(rk&~28)|((rk&4)<<1)|((rk&8)<<1)|((rk&16)>>2)
 FULL:tl.constexpr=2304//DEPTH
 COUNT:tl.constexpr=(4 if DEPTH==512 else 8)
 start=(tl.program_id(0)*BM//TILE)%COUNT
 total=tl.zeros((BM,32),tl.float32)
 for part in tl.static_range(LSU):
  acc=tl.zeros((BM,32),tl.float32)
  for block in range(FULL):
   for sub in tl.static_range(DEPTH//LSU//64):
    kk=((block+start)%FULL)*DEPTH+part*(DEPTH//LSU)+sub*64+rr
    x=tl.load(X+rm[:,None]*2304+kk[None,:],mask=rm[:,None]<M,other=0)
    w=tl.load(W+rn[None,:]*2304+kk[:,None],mask=rn[None,:]<N,other=0)
    acc=tl.dot(x,w,acc)
  if 2304%DEPTH:
   for sub in tl.static_range(DEPTH//LSU//64):
    kk=FULL*DEPTH+part*(DEPTH//LSU)+sub*64+rr
    x=tl.load(X+rm[:,None]*2304+kk[None,:],mask=(rm[:,None]<M)&(kk[None,:]<2304),other=0)
    w=tl.load(W+rn[None,:]*2304+kk[:,None],mask=(rn[None,:]<N)&(kk[:,None]<2304),other=0)
    acc=tl.dot(x,w,acc)
  total=total+acc
 tl.store(Y+rm[:,None]*N+rn[None,:],total.to(tl.bfloat16),mask=(rm[:,None]<M)&(rn[None,:]<N))
r=Path('/workspace/logs/step0_continuation/norm990_active_prod');hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors'];iso=torch.load(r/'ck_stage_isolation.pt',weights_only=False);h=hf['L0.full.post_norm.out0'].reshape(990,2304).cuda();model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');wm=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];rows=[]
profile=json.loads((r/'hf_grouped_gateup_profile_summary.json').read_text());active=iso['ids'].flatten().bincount(minlength=64).nonzero().flatten().tolist();km=dict(zip(active,[k['name'] for k in profile['kernels']]))
for e in active:
 mask=iso['ids']==e;tokens=mask.nonzero()[:,0].cuda();x=h[tokens].contiguous();ref=torch.cat([iso['gate_hf'][mask],iso['up_hf'][mask]],-1).cuda();weights=[]
 for proj in ['gate_proj','up_proj']:
  key=f'model.layers.0.mlp.experts.{e}.{proj}.weight'
  with safe_open(model/wm[key],framework='pt',device='cpu') as f:weights.append(f.get_tensor(key))
 w=torch.cat(weights).cuda();m=x.shape[0];direct=int((torch.nn.functional.linear(x,w)!=ref).sum());mi32='MI32x32' in km[e];dims=__import__('re').search(r'_MT(\d+)x(\d+)x(\d+)_',km[e]);tile=int(dims[2]);depth=int(dims[3]);
 if km[e].endswith('_1'):
  row={'expert':e,'m':m,'unsupported_kernel':km[e]};rows.append(row);print(row,flush=True);continue
 lsu=int(km[e].rsplit('_',1)[1]);bm=32 if mi32 else 16
 for perm in [1 if mi32 else 3]:
  out=torch.empty_like(ref);gateup[(tr.cdiv(m,bm),56)](x,w,out,m,1792,perm,tile,depth,lsu,bm,num_warps=1,matrix_instr_nonkdim=32 if mi32 else 16,enable_fp_fusion=False);row={'expert':e,'m':m,'tile':tile,'mi':32 if mi32 else 16,'perm':perm,'direct_hf_different':direct,'gate_different':int((out[:,:896]!=ref[:,:896]).sum()),'up_different':int((out[:,896:]!=ref[:,896:]).sum())};rows.append(row);print(row,flush=True)
(r/'local_split_korder_profiles.json').write_text(json.dumps(rows,indent=2)+'\n')
