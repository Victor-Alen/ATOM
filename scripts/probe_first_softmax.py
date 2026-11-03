import torch,json
from pathlib import Path
r=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');c=torch.load(r/'prefill_candidate.pt',weights_only=False);s=c['ref_scores'].cuda();n=s.shape[-1];mask=torch.arange(n,device='cuda')[:,None]>=torch.arange(n,device='cuda')[None,:];masked=s.masked_fill(~mask,float('-inf'));p=masked.softmax(-1,dtype=torch.float32).bfloat16();old=c['ref_p'].cuda();d=(p!=old);print('SOFTMAX BF16->FP32 versus explicit cast',int(d.sum()),flush=True)
cp=c['p'].cuda();coords=(cp!=p).nonzero();coords=coords[(coords[:,1]*32*n+coords[:,0]*n+coords[:,2]).argsort()];print('FIRST',coords[:8].tolist(),flush=True)
h,q,k=coords[0].tolist();row=masked[h,q:q+1].float();print('ROW',h,q,k,'BF16 probs',float(cp[h,q,k]),float(p[h,q,k]),'F32 ref',float(row.softmax(-1)[0,k]),flush=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 z=masked.float().softmax(-1);torch.cuda.synchronize()
prof.export_chrome_trace(str(r/'softmax_profile.json'));e=json.loads((r/'softmax_profile.json').read_text())['traceEvents'];print('KERNELS',sorted(set(x['name'] for x in e if x.get('cat')=='kernel')),flush=True)
