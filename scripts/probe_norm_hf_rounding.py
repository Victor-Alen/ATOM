#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
from aiter import rmsnorm2d_fwd,rmsnorm2d_fwd_with_add

ROOT=Path('/workspace/logs/step0_continuation/first_divergence')
def metric(a,b):
    d=(a.float().cpu()-b.float().cpu()).abs()
    return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    a=torch.load(ROOT/'atom/step_27.pt',map_location='cpu',weights_only=False)['tensors']
    rows=[]
    for tag in [f'L{i}.{name}' for i in range(28) for name in ('input_norm','post_norm')]+['final_norm']:
        x=a[tag+'.in0'].cuda();w=a[tag+'.weight'].cuda();r=a.get(tag+'.in1')
        if r is None:
            out=rmsnorm2d_fwd(x,w,1e-6);ref_input=x;res=None
        else:
            r=r.cuda();out=torch.empty_like(x);res=torch.empty_like(x)
            rmsnorm2d_fwd_with_add(out,x,r,res,w,1e-6);ref_input=x+r
        f=ref_input.float();ref=w*(f*torch.rsqrt(f.pow(2).mean(-1,keepdim=True)+1e-6)).to(x.dtype)
        row={'tag':tag,'output_vs_hf':metric(out,ref),'output_vs_captured_baseline':metric(out,a[tag+'.out0'])}
        if res is not None:row['residual_vs_hf']=metric(res,ref_input)
        rows.append(row)
    result={'nodes':len(rows),'output_different':sum(r['output_vs_hf']['different'] for r in rows),
            'output_total':sum(r['output_vs_hf']['total'] for r in rows),
            'residual_different':sum(r.get('residual_vs_hf',{}).get('different',0) for r in rows),
            'capture_different':sum(r['output_vs_captured_baseline']['different'] for r in rows),'rows':rows}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':main()
