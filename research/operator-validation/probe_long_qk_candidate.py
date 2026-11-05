"""Validate generated shape profiles independently of public prompt contents."""
from pathlib import Path
import json, os, sys, gc, re, time
import torch

r=Path('/workspace/logs/step0_operator_parallel_20261104/startup_delivery')
sys.path[:0]=['/workspace/ATOM-csig-final','/workspace/aiter-mellum-head-parallel']
from atom.model_ops.mellum_prefill_sparse_qk import qk, try_qk
config=json.loads((r/'native_sparse_qk_long_config.json').read_text())
classes=json.loads((r/'native_sparse_qk_long_classes.json').read_text())['classes']
os.environ['ATOM_MELLUM_PREFILL_QK_CONFIG']=str(r/'native_sparse_qk_long_config.json')
torch.set_num_threads(2);torch.manual_seed(20261106)
cases=[];started=time.time()
def save(done):
    (r/'long_qk_operator_validation.json').write_text(json.dumps(dict(complete=done,passed=done and all(x.get('passed') for x in cases),cases=cases,elapsed_s=time.time()-started,scope='Random contiguous Q and K; exhaustive shape-class boundary representatives; valid causal SWA region only. Not full-model validation.'),indent=2))
def diff_count(got,expected,n):
    count=0
    for start in range(0,n,256):
        stop=min(start+256,n)
        first=max(0,start-1023)
        qq=torch.arange(start,stop,device='cuda')[:,None]
        kk=torch.arange(first,stop,device='cuda')[None,:]
        mask=(kk<=qq)&(kk>qq-1024)
        count+=int(((got[:,:,start:stop,first:stop]!=expected[:,:,start:stop,first:stop]) & mask).sum())
    return count
with torch.inference_mode():
    for key,lengths in classes.items():
        profile=json.loads(key)
        selected=sorted(set([lengths[0],lengths[min(1,len(lengths)-1)],lengths[len(lengths)//2],lengths[-1]]))
        for n in selected:
            row=dict(n=n,profile=profile);cases.append(row);save(False)
            query=torch.randn((n,32,128),device='cuda',dtype=torch.bfloat16)
            key_tensor=torch.randn((n,4,128),device='cuda',dtype=torch.bfloat16)
            qview=query.transpose(0,1)[None]
            expanded=key_tensor.transpose(0,1)[None,:,None,:,:].expand(1,4,8,n,128).reshape(1,32,n,128)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                expected=torch.matmul(qview,expanded.transpose(2,3));torch.cuda.synchronize()
            row['reference_kernels']=[e.name for e in prof.events() if e.device_type.name=='CUDA' and '_MT' in e.name]
            got=try_qk(query,key_tensor)
            assert got is not None
            row['different']=diff_count(got,expected,n)
            row['passed']=row['different']==0
            # Forty repetitions only at the last member of each shape class.
            if row['passed'] and n==selected[-1]:
                row['repeat_differences']=[]
                for repeat in range(40):
                    got=qk(query,key_tensor,**profile)
                    row['repeat_differences'].append(diff_count(got,expected,n))
                row['passed']=not any(row['repeat_differences'])
            save(False);print('CASE',json.dumps(row),flush=True)
            del query,key_tensor,qview,expanded,expected,got,prof;gc.collect();torch.cuda.empty_cache()
            if not row['passed']:raise RuntimeError(f'QK profile failed at length {n}')
save(True);print('LONG_QK_OPERATOR_PASS',len(cases),flush=True)
