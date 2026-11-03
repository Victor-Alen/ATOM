"""Freeze actual production h/w/router, replay unchanged CK and isolate slot order."""
import torch,json,hashlib,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiter import ActivationType
import atom.models.step_tensor_trace as trace
from atom.model_ops.mellum_ordered_moe import mellum_ordered_moe
tag=os.environ.get('MELLUM_REPLAY_TAG','');root=Path('/workspace/logs/step0_continuation/norm990_active_prod');a=torch.load(root/'atom/step_0.pt',weights_only=False)['tensors'];hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors'];wp=torch.load(root/'atom/L0_weights.pt',weights_only=False)
w13=wp['w13'].cuda();w2=wp['w2'].cuda();w13.is_shuffled=wp['w13_shuffled'];w2.is_shuffled=wp['w2_shuffled'];h=a['L0.full.routing.input'].cuda();ids=a['L0.full.routing.ids'].cuda();rw=a['L0.full.routing.weights'].cuda();hi=hf['L0.full.router.out2'].reshape_as(ids).cuda().to(ids.dtype);hw=hf['L0.full.router.out1'].reshape_as(rw).cuda().float()
captured={};trace.record_moe=lambda *args:None;trace.record_moe_weights=lambda *args:None;trace.record_moe_intermediate=lambda layer,name,value:captured.__setitem__(name,value.clone())
def call(i,r):return mellum_ordered_moe(h,w13,w2,r,i,ActivationType.Silu,'model.layers.0.mlp.experts')
def metric(x,y):
 d=(x.float()-y.to(x.device).reshape_as(x).float()).abs();return {'different':int((d!=0).sum()),'max':float(d.max()),'mean':float(d.mean()),'total':d.numel()}
with torch.inference_mode():
 prod=call(ids,rw);s1=captured['stage1'].clone();s2=captured['stage2_weighted_slots'].clone()
 reference=hf['L0.full.mlp.out0'].reshape_as(prod).cuda();report={'replay_vs_actual_output':metric(prod,a['L0.full.mlp.out0']),'stage1_replay_vs_actual':metric(s1,a['L0.full.moe.stage1']),'stage2_replay_vs_actual':metric(s2,a['L0.full.moe.stage2_weighted_slots']),'production_routes_vs_hf':metric(prod,reference)}
 forced=call(hi,hw);fs1=captured['stage1'].clone();fs2=captured['stage2_weighted_slots'].clone();report['hf_routes_vs_production_routes']=metric(forced,prod);report['hf_routes_vs_hf']=metric(forced,reference)
 # Canonicalize each token's slots by expert to separate row order from arithmetic.
 order=ids.argsort(-1);horder=hi.argsort(-1)
 report['stage1_by_expert_routes_diff']=metric(fs1.gather(1,horder[:,:,None].expand_as(fs1)),s1.gather(1,order[:,:,None].expand_as(s1)))
 report['stage2_by_expert_routes_diff']=metric(fs2.gather(1,horder[:,:,None].expand_as(fs2)),s2.gather(1,order[:,:,None].expand_as(s2)))
 report['production_routes_40_repeats_equal']=all(torch.equal(call(ids,rw),prod) for _ in range(40))
 report['hf_routes_40_repeats_equal']=all(torch.equal(call(hi,hw),forced) for _ in range(40))
 report['HIP_VISIBLE_DEVICES']=os.environ.get('HIP_VISIBLE_DEVICES')
 report['output_sha256']=hashlib.sha256(prod.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
 if tag:
  prior=torch.load(root/'fixed_router_replay.pt',weights_only=False)
  report['production_vs_prior_gpu']=metric(prod,prior['production'])
  report['hf_routes_vs_prior_gpu']=metric(forced,prior['hf_routes'])

 torch.save({'production':prod.cpu(),'hf_routes':forced.cpu(),'stage1_hf_routes':fs1.cpu(),'stage2_hf_routes':fs2.cpu()},root/('fixed_router_replay'+tag+'.pt'))
 (root/('fixed_router_replay'+tag+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
