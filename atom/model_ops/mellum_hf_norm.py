"""Opt-in HF-profiled BF16 RMSNorm for M990,N2304; one real GPU kernel."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.hip import libdevice

@tr.jit
def _norm(X,R,W,Y,RO,Stats,XS:tl.constexpr,RS:tl.constexpr,EPS:tl.constexpr,ADD:tl.constexpr):
 row=tl.program_id(0);lane=tl.arange(0,64)
 a0=tl.full((64,),0,tl.float32);a1=tl.full((64,),0,tl.float32);a2=tl.full((64,),0,tl.float32);a3=tl.full((64,),0,tl.float32);values=()
 for it in tl.static_range(9):
  d=it*256+lane*4
  x0=tl.load(X+row*XS+d).to(tl.float32);x1=tl.load(X+row*XS+d+1).to(tl.float32);x2=tl.load(X+row*XS+d+2).to(tl.float32);x3=tl.load(X+row*XS+d+3).to(tl.float32)
  if ADD:
   x0=(x0+tl.load(R+row*RS+d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
   x1=(x1+tl.load(R+row*RS+d+1).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
   x2=(x2+tl.load(R+row*RS+d+2).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
   x3=(x3+tl.load(R+row*RS+d+3).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
   tl.store(RO+row*2304+d,x0);tl.store(RO+row*2304+d+1,x1);tl.store(RO+row*2304+d+2,x2);tl.store(RO+row*2304+d+3,x3)
  values+=(x0,x1,x2,x3)
  a0=a0+x0*x0;a1=a1+x1*x1;a2=a2+x2*x2;a3=a3+x3*x3
 acc=((a0+a1)+a2)+a3
 # ATen Reduce.cuh ROCm block_x_reduce uses increasing shfl_down offsets.
 for level in tl.static_range(6):
  off=1<<level;src=tl.where(lane+off<64,lane+off,lane)
  acc=acc+tl.gather(acc,src,0)
 total=tl.sum(tl.where(lane==0,acc,0),0)
 # Match the observed HF/ROCm scalar rsqrt result before returning to FP32.
 mean=total*(1.0/2304);var=mean+EPS;inv=libdevice.rsqrt(var.to(tl.float64)).to(tl.float32)
 tl.store(Stats+row*3,total);tl.store(Stats+row*3+1,var);tl.store(Stats+row*3+2,inv)
 for it in tl.static_range(9):
  for j in tl.static_range(4):
   d=it*256+lane*4+j;z=(values[it*4+j]*inv).to(tl.bfloat16).to(tl.float32);y=z*tl.load(W+d).to(tl.float32)
   tl.store(Y+row*2304+d,y.to(tl.bfloat16))

def rmsnorm_hf_prefill(x,w,eps=1e-6,residual=None,*,return_stats=False):
 assert x.shape==(990,2304) and x.dtype==w.dtype==torch.bfloat16 and x.stride(-1)==1
 if residual is not None:assert residual.shape==x.shape and residual.dtype==x.dtype and residual.stride(-1)==1
 out=torch.empty_like(x);rout=torch.empty_like(x) if residual is not None else None;stats=torch.empty((990,3),device=x.device,dtype=torch.float32)
 _norm[(990,)](x,residual if residual is not None else x,w,out,rout if rout is not None else x,stats,x.stride(0),residual.stride(0) if residual is not None else 0,eps,residual is not None,num_warps=1,enable_fp_fusion=False)
 if __import__('os').getenv('ATOM_STEP_TRACE_DIR'):
  from atom.models.step_tensor_trace import RECORDER
  if RECORDER is not None and RECORDER.active:
   count=RECORDER.detail.get('norm.hf_prefill.count',0)
   RECORDER.detail['norm.hf_prefill.count']=count+1
   if count<2:RECORDER.detail[f'norm.hf_prefill.{count}.stats']=stats.detach().cpu().clone()
 return (out,rout,stats) if return_stats else ((out,rout) if rout is not None else out)
