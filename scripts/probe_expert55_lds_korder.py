import torch,triton as tr,triton.language as tl,json
from pathlib import Path
from safetensors import safe_open
@tr.jit
def gateup(X,W,Y,M:tl.constexpr,N:tl.constexpr,STAGGER:tl.constexpr,PERM:tl.constexpr):
 rm=tl.program_id(0)*32+tl.arange(0,32);rn=tl.program_id(1)*32+tl.arange(0,32);rk=tl.arange(0,64)
 if STAGGER==1:start=(tl.program_id(0)*32//128)%16
 elif STAGGER==2:start=(tl.program_id(1)*32//192)%16
 else:start=0
 acc=tl.zeros((32,32),tl.float32)
 for chunk in range(36):
  if PERM:rr=(rk&~12)|((rk&4)<<1)|((rk&8)>>1)
  else:rr=rk
  kk=((chunk+start)%36)*64+rr
  x=tl.load(X+rm[:,None]*2304+kk[None,:],mask=rm[:,None]<M,other=0)
  w=tl.load(W+rn[None,:]*2304+kk[:,None],mask=rn[None,:]<N,other=0)
  acc=tl.dot(x,w,acc)
 tl.store(Y+rm[:,None]*N+rn[None,:],acc.to(tl.bfloat16),mask=(rm[:,None]<M)&(rn[None,:]<N))
r=Path('/workspace/logs/step0_continuation/norm990_active_prod');hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors'];iso=torch.load(r/'ck_stage_isolation.pt',weights_only=False);h=hf['L0.full.post_norm.out0'].reshape(990,2304).cuda();mask=iso['ids']==55
ref=torch.cat([iso['gate_hf'][mask],iso['up_hf'][mask]],-1).cuda();ckg=iso['gate_ck'][mask].cuda();model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');wm=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];weights=[]
for name in ['gate_proj','up_proj']:
 key='model.layers.0.mlp.experts.55.'+name+'.weight'
 with safe_open(model/wm[key],framework='pt',device='cpu') as f:weights.append(f.get_tensor(key))
w=torch.cat(weights).cuda();rows=[]
print('DIRECT_F_LINEAR_VS_GROUPED_HF',int((torch.nn.functional.linear(h,w)!=ref).sum()),flush=True)
def metric(a,b):
 d=(a.float()-b.float()).abs();return {'different':int((d!=0).sum()),'max':float(d.max()),'mean':float(d.mean()),'total':d.numel()}
for mfma in [16,32]:
 for stagger in [0,1]:
  for perm in [False,True]:
   out=torch.empty((990,1792),device='cuda',dtype=torch.bfloat16);compiled=gateup[(tr.cdiv(990,32),tr.cdiv(1792,32))](h,w,out,990,1792,stagger,perm,num_warps=1,matrix_instr_nonkdim=mfma,enable_fp_fusion=False)
   rows.append({'mfma':mfma,'swap_k_bits_2_3':perm,'stagger':stagger,'gate_vs_hf':metric(out[:,:896],ref[:,:896]),'up_vs_hf':metric(out[:,896:],ref[:,896:]),'gate_vs_ck16':metric(out[:,:896],ckg),'first_gate_0_407':float(out[0,407]),'instructions':sorted(set(s.strip().split(' ')[0] for s in compiled.asm['amdgcn'].splitlines() if 'v_mfma_' in s))})
(r/'expert55_lds_korder_probe.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
