from pathlib import Path
import json,os,sys,time,gc
r=Path('/workspace/logs/step0_operator_parallel_20261104/startup_delivery')
base=json.loads(Path('/workspace/ATOM-csig-final/scripts/server_profiles/fast.json').read_text())
os.environ.update(base['env']);os.environ.update(HIP_VISIBLE_DEVICES='1',ROCR_VISIBLE_DEVICES='0,1',TRITON_CACHE_DIR=str(r/'pv_triton'))
sys.path[:0]=['/workspace/ATOM-csig-final','/workspace/aiter-mellum-head-parallel',str(r)]
import torch
from mellum_prefill_sparse_pv import pv
torch.set_num_threads(2);torch.manual_seed(20261106)
results=[];started=time.time()
def save(done=False):
    (r/'pv_operator_probe.json').write_text(json.dumps(dict(complete=done,cases=results,elapsed_s=time.time()-started),indent=2))
with torch.inference_mode():
    for n in [4096,4097,7169,16384]:
        p=torch.zeros((1,32,n,n),device='cuda',dtype=torch.bfloat16)
        for start in range(0,n,128):
            stop=min(n,start+128);left=max(0,start-1023)
            rr=torch.arange(start,stop,device='cuda')[:,None];kk=torch.arange(left,stop,device='cuda')[None,:]
            values=torch.rand((1,32,stop-start,stop-left),device='cuda',dtype=torch.bfloat16)
            values*=((kk<=rr)&(kk>rr-1024))
            p[:,:,start:stop,left:stop]=values
        v=torch.randn((1,32,n,128),device='cuda',dtype=torch.bfloat16)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            ref=p@v;torch.cuda.synchronize()
        kernels=[e.name for e in prof.events() if e.device_type.name=='CUDA' and '_MT' in e.name]
        for kw,rotate in [(8,True),(4,True),(8,False),(4,False)]:
            got=pv(p,v,kw=kw,rotate=rotate)
            diff=int((got!=ref).sum());maxdiff=float((got.float()-ref.float()).abs().max())
            row=dict(n=n,kw=kw,rotate=rotate,different=diff,max_abs=maxdiff,kernels=kernels)
            results.append(row);save();print('PV',json.dumps(row),flush=True)
            if diff==0:break
        del p,v,ref,got,prof;gc.collect();torch.cuda.empty_cache()
save(True)
