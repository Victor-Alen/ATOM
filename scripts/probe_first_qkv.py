import json,torch,os
from pathlib import Path
from safetensors import safe_open
from aiter.tuned_gemm import get_GEMM_A16W16_config,tgemm,hipb_gemm
root=Path(os.environ.get('MELLUM_TRACE_ROOT','/workspace/logs/step0_continuation/hf_attn_first29'));target=int(os.environ.get('MELLUM_TRACE_STEP','29'));a=torch.load(root/'atom'/f'step_{target}.pt',weights_only=False,map_location='cpu')['tensors'];model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');wm=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
def weight(n):
 n='model.layers.0.self_attn.'+n
 with safe_open(model/wm[n],framework='pt',device='cpu') as f:return f.get_tensor(n).cuda()
def metric(x,y):
 d=(x.float()-y.float()).abs();return {'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'total':d.numel()}
w=[weight(n+'_proj.weight') for n in ['q','k','v']];joined=torch.cat(w);report={'steps':[]}
with torch.inference_mode():
 for step in sorted(set([0,target])):
  x=a[f'L0.step{step}.qkv_proj.in0'].cuda();actual=a[f'L0.step{step}.qkv_proj.out0'].cuda();hf=torch.cat([torch.nn.functional.linear(x,z) for z in w],-1)
  replay=tgemm.mm(x,joined,otype=torch.bfloat16)
  separate=torch.cat([tgemm.mm(x,z,otype=torch.bfloat16) for z in w],-1)
  cfg=get_GEMM_A16W16_config(x.shape[0],5120,2304,False,'torch.bfloat16','torch.bfloat16')
  coords=(actual!=hf).nonzero();row={'step':step,'config':cfg,'actual_vs_hf':metric(actual,hf),'replay_vs_actual':metric(replay,actual),'separate_tgemm_vs_hf':metric(separate,hf),'first_differences':[]}
  for tok,col in coords[:10].tolist():
   exact=float((x[tok].double()*joined[col].double()).sum())
   row['first_differences'].append({'token':tok,'column':col,'hf':float(hf[tok,col]),'actual':float(actual[tok,col]),'fp64_dot':exact,'midpoint':(float(hf[tok,col])+float(actual[tok,col]))/2})
  report['steps'].append(row)
 (root/'first_qkv.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
