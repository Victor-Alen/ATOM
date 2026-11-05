from pathlib import Path
import json,os,sys,time,gc
r=Path('/workspace/logs/step0_operator_parallel_20261104/startup_delivery')
base=json.loads(Path('/workspace/ATOM-csig-final/scripts/server_profiles/fast.json').read_text())
os.environ.update(base['env']);os.environ.update(HIP_VISIBLE_DEVICES='1',ROCR_VISIBLE_DEVICES='0,1',TRITON_CACHE_DIR=str(r/'window_sparse_triton'))
sys.path[:0]=['/workspace/ATOM-csig-final','/workspace/aiter-mellum-head-parallel',str(r)]
import torch,triton
from mellum_prefill_softmax_window_sparse import softmax as candidate
from mellum_prefill_sparse_pv_masked import pv
from atom.model_ops.mellum_prefill_softmax_gluon import softmax as accepted
torch.set_num_threads(2);torch.manual_seed(20261106)
results=[];started=time.time()
def save(done=False):
    (r/'window_sparse_validation.json').write_text(json.dumps(dict(complete=done,passed=done and all(x.get('different')==0 and x.get('pv_different')==0 for x in results),cases=results,elapsed_s=time.time()-started),indent=2))
with torch.inference_mode():
    for n in [3073,4095,4096,4097,7168,7169,10001,12580,15194,16383,16384]:
        scores=torch.empty((1,32,n,n),device='cuda',dtype=torch.bfloat16)
        for start in range(0,n,128):
            stop=min(n,start+128);left=max(0,start-1023)
            scores[:,:,start:stop,left:stop]=torch.randn((1,32,stop-start,stop-left),device='cuda',dtype=torch.bfloat16)*5
        ref=accepted(scores,n,n,1024,128**-0.5)
        got=candidate(scores,n,n,1024,128**-0.5,rows=1,sparse_output=True)
        diff=0
        for start in range(0,n,128):
            stop=min(n,start+128);left=max(0,start-1023)
            rr=torch.arange(start,stop,device='cuda')[:,None];kk=torch.arange(left,stop,device='cuda')[None,:]
            mask=(kk<=rr)&(kk>rr-1024)
            diff+=int(((got[:,:,start:stop,left:stop]!=ref[:,:,start:stop,left:stop])&mask).sum())
        v=torch.randn((1,32,n,128),device='cuda',dtype=torch.bfloat16)
        expected=ref@v;result=pv(got,v)
        row=dict(n=n,different=diff,pv_different=int((result!=expected).sum()))
        if n in [4096,7169,16384] and diff==0 and row['pv_different']==0:
            row['accepted_softmax_ms']=triton.testing.do_bench(lambda:accepted(scores,n,n,1024,128**-0.5),warmup=50,rep=200)
            row['sparse_softmax_ms']=triton.testing.do_bench(lambda:candidate(scores,n,n,1024,128**-0.5,rows=1,sparse_output=True),warmup=50,rep=200)
        results.append(row);save();print('WINDOW',json.dumps(row),flush=True)
        assert diff==0 and row['pv_different']==0
        del scores,ref,got,v,expected,result;gc.collect();torch.cuda.empty_cache()
save(True);print('WINDOW_SPARSE_PASS',flush=True)
