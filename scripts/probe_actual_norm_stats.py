import torch,json,ctypes
from pathlib import Path
from aiter import rmsnorm2d_fwd_with_add
r=Path('/workspace/logs/step0_continuation/prefill990_prod');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];stem='L0.full.post_norm';x=a[stem+'.in0'].cuda();res=a[stem+'.in1'].cuda();w=a['L0.post_norm.weight'].cuda();out=torch.empty_like(x);rout=torch.empty_like(x)
rmsnorm2d_fwd_with_add(out,x,res,rout,w,1e-6);torch.cuda.synchronize()
lib=ctypes.CDLL('/workspace/aiter-mellum-norm-stats-jit/module_rmsnorm_quant.so');fn=lib.mellum_copy_norm_stats;fn.argtypes=[ctypes.c_void_p,ctypes.c_int];fn.restype=ctypes.c_int;buf=torch.empty((x.shape[0],3),dtype=torch.float32);assert fn(buf.data_ptr(),x.shape[0])==0
f=(x+res).float();mean=f.square().mean(-1);var=mean+1e-6;rsqrt=var.rsqrt();candidate=out.cpu();prior=a[stem+'.out0'];different=int((candidate!=prior).sum());rows=[]
for t in [146]+(buf[:,2]!=rsqrt.cpu()).nonzero().flatten().tolist()[:20]:
 rows.append({'token':t,'kernel_sum':float(buf[t,0]),'kernel_variance':float(buf[t,1]),'kernel_rsqrt':float(buf[t,2]),'hf_mean':float(mean[t]),'hf_variance':float(var[t]),'hf_rsqrt':float(rsqrt[t]),'mean_from_kernel_sum':float(buf[t,0]/2304)})
report={'debug_kernel_vs_original_different':different,'variance_different':int((buf[:,1]!=var.cpu()).sum()),'rsqrt_different':int((buf[:,2]!=rsqrt.cpu()).sum()),'rows':rows};(r/'norm_actual_stats.json').write_text(json.dumps(report,indent=2)+'\n');torch.save({'stats':buf,'hf_mean':mean.cpu(),'hf_var':var.cpu(),'hf_rsqrt':rsqrt.cpu()},r/'norm_actual_stats.pt');print(json.dumps(report,indent=2))
