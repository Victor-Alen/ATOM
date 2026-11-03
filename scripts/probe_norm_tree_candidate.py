import torch,json,importlib.util
from pathlib import Path
r=Path('/workspace/logs/step0_continuation/prefill990_prod');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors'];spec=importlib.util.spec_from_file_location('newnorm','/workspace/ATOM-tritonmoe/atom/model_ops/mellum_hf_norm.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def metric(a,b):
 a=a.reshape(-1).float();b=b.reshape(-1).to(a.device).float();d=(a-b).abs();return {'different':int((d!=0).sum()),'max':float(d.max()),'mean':float(d.mean()),'total':d.numel()}
rows=[]
for name in ['input_norm','post_norm']:
 stem='L0.full.'+name;x=a[stem+'.in0'].cuda();res=a.get(stem+'.in1');res=None if res is None else res.cuda();w=a['L0.'+name+'.weight'].cuda();out,ro,stats=m.rmsnorm_hf_prefill(x,w,residual=res,return_stats=True);actualhf=hf[stem+'.out0'].cuda();summed=x if res is None else x+res;f=summed.float();mean=f.square().mean(-1);var=mean+1e-6;iv=var.rsqrt()
 row={'name':name,'actual_hf_input_diff':metric(summed,hf[stem+'.in0']),'new_vs_actual_hf':metric(out,actualhf),'baseline_vs_actual_hf':metric(a[stem+'.out0'],actualhf.cpu()),'variance_vs_hf':metric(stats[:,1],var),'rsqrt_vs_hf':metric(stats[:,2],iv),'row146':{'sum':float(stats[146,0]),'variance':float(stats[146,1]),'rsqrt':float(stats[146,2])}}
 if ro is not None:row['residual_vs_hf_input']=metric(ro,hf[stem+'.in0'])
 rows.append(row)
(r/'norm_tree_candidate.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
