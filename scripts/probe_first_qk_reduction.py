import torch,json
from pathlib import Path
root=Path('/workspace/logs/step0_continuation/qkv_actual_rope_prefill');a=torch.load(root/'atom/step_0.pt',weights_only=False)['tensors'];c=torch.load(root/'prefill_candidate.pt',weights_only=False)
q=a['L0.step0.attn.afterquery'].reshape(-1,32,128).transpose(0,1).cuda();k=a['L0.step0.attn.afterkey'].reshape(-1,4,128).transpose(0,1).repeat_interleave(8,0).cuda();n=q.shape[1]
cs=c['scores'];rs=c['ref_scores'];coords=(cs!=rs).nonzero();coords=coords[coords[:,2]<=coords[:,1]];order=(coords[:,1]*32*n+coords[:,0]*n+coords[:,2]).argsort();coords=coords[order]
rows=[];raw=q@k.transpose(-1,-2)
for h,t,j in coords[:20].tolist():
 exact=float((q[h,t].double()*k[h,j].double()).sum());rows.append({'head':h,'query':t,'key':j,'qk_fp64':exact,'qk_hf_bf16':float(raw[h,t,j]),'scaled_hf':float(rs[h,t,j]),'scaled_triton':float(cs[h,t,j])})
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 out=q@k.transpose(-1,-2);torch.cuda.synchronize()
prof.export_chrome_trace(str(root/'qk_profile.json'));events=json.loads((root/'qk_profile.json').read_text())['traceEvents'];names=sorted(set(e.get('name','') for e in events if e.get('cat')=='kernel'))
r={'different_valid_scaled_scores':len(coords),'first20':rows,'torch_qk_kernel_names':names};(root/'first_qk_reduction.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
