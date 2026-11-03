import json,torch
from pathlib import Path
from aiter import rmsnorm2d_fwd_with_add
r=Path('/workspace/logs/step0_continuation/prefill990_prod');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];b=torch.load(r/'norm_actual_stats.pt',weights_only=False);x=a['L0.full.post_norm.in0'].cuda();res=a['L0.full.post_norm.in1'].cuda();w=a['L0.post_norm.weight'].cuda();f=(x+res).float();sq=f.square();s=sq.sum(-1);mean=sq.mean(-1);stats=b['stats'].cuda();factor=torch.tensor(1/2304,device='cuda',dtype=torch.float32)
variants={'hf_sum':s,'kernel_sum':stats[:,0],'hf_mean':mean,'hf_sum_times_factor':s*factor,'kernel_sum_times_factor':stats[:,0]*factor,'kernel_variance':stats[:,1],'hf_variance':mean+1e-6,'kernel_sum_hf_variance':stats[:,0]*factor+1e-6,'kernel_sum_double_epsilon':((stats[:,0]/2304).double()+1e-6).float(),'hf_rsqrt_on_kernel_variance':stats[:,1].rsqrt(),'kernel_rsqrt':stats[:,2]}
def neq(a,b):return int((a!=b).sum())
report={'hf_sum_times_factor_vs_mean':neq(s*factor,mean),'kernel_sum_vs_hf_sum':neq(stats[:,0],s),'kernel_var_vs_hf_math_same_sum':neq(stats[:,1],variants['kernel_sum_hf_variance']),'kernel_var_vs_double_epsilon':neq(stats[:,1],variants['kernel_sum_double_epsilon']),'kernel_rsqrt_vs_hf_same_var':neq(stats[:,2],stats[:,1].rsqrt()),'token146':{k:float(v[146]) for k,v in variants.items()}}
out=torch.empty_like(x);ro=torch.empty_like(x)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 z=sq.mean(-1);zz=z.rsqrt();rmsnorm2d_fwd_with_add(out,x,res,ro,w,1e-6);torch.cuda.synchronize()
prof.export_chrome_trace(str(r/'norm_profile.json'));e=json.loads((r/'norm_profile.json').read_text())['traceEvents'];report['kernels']=[{'name':v['name'],'launch':v.get('args')} for v in e if v.get('cat')=='kernel'];(r/'norm_stage_isolation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
