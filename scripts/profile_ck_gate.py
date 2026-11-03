import sys,torch,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiter import ActivationType,QuantType
from aiter.fused_moe import get_2stage_cfgs,get_padded_M,moe_sorting
r=Path('/workspace/logs/step0_continuation/norm990_active_prod');a=torch.load(r/'atom/step_0.pt',weights_only=False)['tensors'];wp=torch.load(r/'atom/L0_weights.pt',weights_only=False);w13=wp['w13'].cuda();w2=wp['w2'].cuda();w13.is_shuffled=wp['w13_shuffled'];w2.is_shuffled=wp['w2_shuffled'];h=a['L0.full.routing.input'].cuda();ids=a['L0.full.routing.ids'].cuda();rw=a['L0.full.routing.weights'].cuda();m=h.shape[0]
meta=get_2stage_cfgs(get_padded_M(m),2304,896,64,8,h.dtype,w13.dtype,w13.dtype,QuantType.No,True,ActivationType.Silu,False,0,0,True);sid,sw,se,nv,_=moe_sorting(ids,rw,64,2304,h.dtype,meta.block_m);out=torch.empty((m,8,896),device='cuda',dtype=h.dtype)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 meta.stage1(h,w13,w2,sid,se,nv,out,8,block_m=meta.block_m,a1_scale=None,w1_scale=None,sorted_weights=None);torch.cuda.synchronize()
prof.export_chrome_trace(str(r/'ck_gate_profile.json'));e=json.loads((r/'ck_gate_profile.json').read_text())['traceEvents'];report={'stage1_keywords':{k:str(v) for k,v in getattr(meta.stage1,'keywords',{}).items()},'kernels':[{'name':x['name'],'launch':x.get('args')} for x in e if x.get('cat')=='kernel']};(r/'ck_gate_profile_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
