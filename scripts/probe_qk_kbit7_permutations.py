import torch,triton as tr,triton.language as tl,itertools,json
from pathlib import Path
@tr.jit
def kernel(Q,K,Perm,O,N:tl.constexpr):
 pid=tl.program_id(0);case=tl.program_id(1);d=tl.arange(0,16);r=tl.arange(0,16)
 p0=tl.load(Perm+pid*7);p1=tl.load(Perm+pid*7+1);p2=tl.load(Perm+pid*7+2);p3=tl.load(Perm+pid*7+3)
 p4=tl.load(Perm+pid*7+4);p5=tl.load(Perm+pid*7+5);p6=tl.load(Perm+pid*7+6)
 acc=tl.zeros((16,16),tl.float32)
 for chunk in tl.static_range(8):
  dd0=chunk*16+d;dd=((dd0&1)<<p0)|(((dd0>>1)&1)<<p1)|(((dd0>>2)&1)<<p2)|(((dd0>>3)&1)<<p3)|(((dd0>>4)&1)<<p4)|(((dd0>>5)&1)<<p5)|(((dd0>>6)&1)<<p6)
  a=tl.load(Q+case*128+dd);b=tl.load(K+case*128+dd)
  aa=tl.broadcast_to(a[None,:],(16,16));bb=tl.broadcast_to(b[:,None],(16,16))
  acc=tl.dot(aa,bb,acc)
 score=(acc.to(tl.bfloat16).to(tl.float32)*(128**-0.5)).to(tl.bfloat16)
 scalar=tl.sum(tl.sum(tl.where((r[:,None]==0)&(r[None,:]==0),score.to(tl.float32),0),0),0);tl.store(O+pid*N+case,scalar.to(tl.bfloat16))
r=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');c=torch.load(r/'prefill_candidate.pt',weights_only=False);a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];q=a['L0.step0.attn.afterquery'].reshape(-1,32,128).transpose(0,1);k=a['L0.step0.attn.afterkey'].reshape(-1,4,128).transpose(0,1).repeat_interleave(8,0);cs=c['scores'];rs=c['ref_scores'];coords=(cs!=rs).nonzero();coords=coords[coords[:,2]<=coords[:,1]];coords=coords[(coords[:,1]*32*990+coords[:,0]*990+coords[:,2]).argsort()]
qq=q[coords[:,0],coords[:,1]].contiguous().cuda();kk=k[coords[:,0],coords[:,2]].contiguous().cuda();perms=list(itertools.permutations(range(7)));pm=torch.tensor(perms,device='cuda',dtype=torch.int32);out=torch.empty((len(perms),len(coords)),device='cuda',dtype=torch.bfloat16)
kernel[(len(perms),len(coords))](qq,kk,pm,out,len(coords),num_warps=4,enable_fp_fusion=False)
small=out.cpu();gold=rs[coords[:,0],coords[:,1],coords[:,2]];prior=cs[coords[:,0],coords[:,1],coords[:,2]]
report={'fixture_identity_vs_original_mismatches':int((small[0]!=prior).sum()),'cases':len(coords),'permutations':[{'bits':list(p),'different_from_hf':int((small[i]!=gold).sum())} for i,p in enumerate(perms)]}
(r/'qk_kbit7_permutations.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({**{k:v for k,v in report.items() if k!='permutations'},'best':sorted(report['permutations'],key=lambda z:z['different_from_hf'])[:12]},indent=2))

