import torch,importlib.util,json
from pathlib import Path
r=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];q=a['L0.step0.attn.afterquery'].cuda().reshape(-1,32,128);k=a['L0.step0.attn.afterkey'].cuda().reshape(-1,4,128);n=q.shape[0]
spec=importlib.util.spec_from_file_location('candidate','/workspace/ATOM-tritonmoe/atom/model_ops/mellum_hf_attention.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
lens=torch.arange(1,n+1,device='cuda',dtype=torch.int32);scores=torch.empty((n,4,16,1024),device='cuda',dtype=torch.bfloat16)
m._qk[(n,4,8)](q,k,lens,lens,scores,*q.stride()[:2],*k.stride(),0,0,0,16,8,8,4,1024,1.0,DENSE=True,num_warps=4,enable_fp_fusion=False)
raw=scores[:,:,:8,:n].permute(1,2,0,3).reshape(32,n,n);ref=q.transpose(0,1)@k.transpose(0,1).repeat_interleave(8,0).transpose(-1,-2);mask=torch.arange(n,device='cuda')[:,None]>=torch.arange(n,device='cuda')[None,:];coords=((raw!=ref)&mask).nonzero();coords=coords[(coords[:,1]*32*n+coords[:,0]*n+coords[:,2]).argsort()]
rows=[]
for h,t,j in coords[:10].tolist():rows.append({'h':h,'q':t,'k':j,'candidate_raw':float(raw[h,t,j]),'hf_raw':float(ref[h,t,j])})
report={'different_valid_raw_qk':len(coords),'first10':rows};(r/'raw_qk.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
