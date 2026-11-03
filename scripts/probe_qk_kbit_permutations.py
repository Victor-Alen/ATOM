import torch,triton as tr,triton.language as tl,itertools,json
from pathlib import Path
@tr.jit
def kernel(Q,K,Perm,O,N:tl.constexpr):
 pid=tl.program_id(0);case=tl.program_id(1);d=tl.arange(0,16);r=tl.arange(0,16)
 p0=tl.load(Perm+pid*4);p1=tl.load(Perm+pid*4+1);p2=tl.load(Perm+pid*4+2);p3=tl.load(Perm+pid*4+3)
 perm=((d&1)<<p0)|(((d>>1)&1)<<p1)|(((d>>2)&1)<<p2)|(((d>>3)&1)<<p3)
 acc=tl.zeros((16,16),tl.float32)
 for chunk in tl.static_range(8):
  dd=chunk*16+perm
  a=tl.load(Q+case*128+dd);b=tl.load(K+case*128+dd)
  aa=tl.broadcast_to(a[None,:],(16,16));bb=tl.broadcast_to(b[:,None],(16,16))
  acc=tl.dot(aa,bb,acc)
 score=(acc.to(tl.bfloat16).to(tl.float32)*(128**-0.5)).to(tl.bfloat16)
 tl.store(O+(pid*N+case)*256+r[:,None]*16+r[None,:],score)
r=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');c=torch.load(r/'prefill_candidate.pt',weights_only=False);a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];q=a['L0.step0.attn.afterquery'].reshape(-1,32,128).transpose(0,1);k=a['L0.step0.attn.afterkey'].reshape(-1,4,128).transpose(0,1).repeat_interleave(8,0);cs=c['scores'];rs=c['ref_scores'];coords=(cs!=rs).nonzero();coords=coords[coords[:,2]<=coords[:,1]];coords=coords[(coords[:,1]*32*990+coords[:,0]*990+coords[:,2]).argsort()]
qq=q[coords[:,0],coords[:,1]].contiguous().cuda();kk=k[coords[:,0],coords[:,2]].contiguous().cuda();perms=list(itertools.permutations(range(4)));pm=torch.tensor(perms,device='cuda',dtype=torch.int32);out=torch.empty((24,len(coords),16,16),device='cuda',dtype=torch.bfloat16)
kernel[(24,len(coords))](qq,kk,pm,out,len(coords),num_warps=4,enable_fp_fusion=False)
small=out[:,:,0,0].cpu();gold=rs[coords[:,0],coords[:,1],coords[:,2]];prior=cs[coords[:,0],coords[:,1],coords[:,2]]
report={'fixture_identity_vs_original_mismatches':int((small[0]!=prior).sum()),'fixture_output_positions_differ':int((out!=out[:,:,:1,:1]).sum()),'cases':len(coords),'permutations':[{'bits':list(p),'different_from_hf':int((small[i]!=gold).sum())} for i,p in enumerate(perms)]}
(r/'qk_kbit_permutations.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
