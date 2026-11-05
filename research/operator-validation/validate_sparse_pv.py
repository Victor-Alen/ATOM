from pathlib import Path
import json,os,sys,time,gc,re
r=Path('/workspace/logs/step0_operator_parallel_20261104/startup_delivery')
base=json.loads(Path('/workspace/ATOM-csig-final/scripts/server_profiles/fast.json').read_text())
os.environ.update(base['env']);os.environ.update(HIP_VISIBLE_DEVICES='1',ROCR_VISIBLE_DEVICES='0,1',TRITON_CACHE_DIR=str(r/'pv_triton'))
sys.path[:0]=['/workspace/ATOM-csig-final','/workspace/aiter-mellum-head-parallel',str(r)]
import torch,triton
from mellum_prefill_sparse_pv import pv,try_pv
torch.set_num_threads(2);torch.manual_seed(20261106)
profiles={}
for line in (r/'hf_pv_all_lengths.tsv').read_text().splitlines():
    n,kernel,ws=line.split('\t');n=int(n)
    if n>=3073:
        assert '_MT128x288x64_MI16x16x1_' in kernel
        profiles[n]=dict(mi=16,kw=8,row_tile=288,depth=64,rotate=False)
config=dict(torch_version=torch.__version__,profiles=profiles,scope='SWA1024 full prefill contiguous BF16 probability and repeated V; shape-only dispatch.')
(r/'native_sparse_pv_config.json').write_text(json.dumps(config,indent=2))
os.environ['ATOM_MELLUM_PREFILL_PV_CONFIG']=str(r/'native_sparse_pv_config.json')
results=[];started=time.time()
def save(done=False):
    (r/'pv_operator_validation.json').write_text(json.dumps(dict(complete=done,passed=done and all(x['different']==0 and not any(x.get('repeats',[])) for x in results),cases=results,elapsed_s=time.time()-started),indent=2))
with torch.inference_mode():
    for n in [3073,3104,4095,4096,4097,7168,7169,10001,12580,15194,16383,16384]:
        p=torch.zeros((1,32,n,n),device='cuda',dtype=torch.bfloat16)
        for start in range(0,n,128):
            stop=min(n,start+128);left=max(0,start-1023)
            rr=torch.arange(start,stop,device='cuda')[:,None];kk=torch.arange(left,stop,device='cuda')[None,:]
            values=torch.randn((1,32,stop-start,stop-left),device='cuda',dtype=torch.float32)*3
            values.masked_fill_(~((kk<=rr)&(kk>rr-1024)),float('-inf'))
            p[:,:,start:stop,left:stop]=values.softmax(-1).to(torch.bfloat16)
        v=torch.randn((1,32,n,128),device='cuda',dtype=torch.bfloat16)
        ref=p@v;got=try_pv(p,v)
        assert got is not None
        diff=int((got!=ref).sum())
        row=dict(n=n,different=diff,max_abs=float((got.float()-ref.float()).abs().max()))
        if diff:
            row['coordinates']=(got!=ref).nonzero()[:30].tolist()
            row['kw4_diff']=int((pv(p,v,kw=4)!=ref).sum())
            row['rotated_diff']=int((pv(p,v,rotate=True)!=ref).sum())
            torch.save(dict(p=p,v=v,ref=ref,got=got),r/'pv_first_failure.pt')
        if n==16384 and diff==0:
            row['repeats']=[int((pv(p,v)!=ref).sum()) for _ in range(40)]
        if n in [4096,7169,16384] and diff==0:
            row['reference_ms']=triton.testing.do_bench(lambda:p@v,warmup=100,rep=300)
            row['candidate_ms']=triton.testing.do_bench(lambda:pv(p,v),warmup=100,rep=300)
        results.append(row);save();print('VALIDATED',json.dumps(row),flush=True)
        assert diff==0 and not any(row.get('repeats',[]))
        del p,v,ref,got;gc.collect();torch.cuda.empty_cache()
save(True);print('PV_OPERATOR_PASS',flush=True)
