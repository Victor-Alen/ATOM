import os,sys,json,hashlib,torch
from pathlib import Path
sys.path.insert(0,'/workspace/ATOM-tritonmoe')
from atom.model_ops.mellum_ordered_moe import mellum_ordered_moe
from atom.model_ops.mellum_ck_korder import ck_hf_korder_plan
from aiter import ActivationType
root=Path('/workspace/logs/step0_continuation/norm990_active_prod');outroot=Path('/workspace/logs/step0_continuation/ck_hf_profile_prod');tag=sys.argv[1]
a=torch.load(root/'atom/step_0.pt',weights_only=False)['tensors'];wp=torch.load(root/'atom/L0_weights.pt',weights_only=False);w13=wp['w13'].cuda();w2=wp['w2'].cuda();w13.is_shuffled=wp['w13_shuffled'];w2.is_shuffled=wp['w2_shuffled'];h=a['L0.full.routing.input'].cuda();ids=a['L0.full.routing.ids'].cuda();rw=a['L0.full.routing.weights'].cuda()
base=torch.load(root/'ck_stage_isolation.pt',weights_only=False);new=torch.load(root/'ck_stage_isolation_profile.pt',weights_only=False);report={'up_target':{'actual':float(new['up_ck'][0,5,783]),'hf':float(base['up_hf'][0,5,783])},'old_gate_target':{'actual':float(new['gate_ck'][0,0,407]),'hf':float(base['gate_hf'][0,0,407])}}
assert report['up_target']['actual']==report['up_target']['hf'];assert report['old_gate_target']['actual']==report['old_gate_target']['hf']
table=torch.tensor(json.loads(Path('/workspace/ATOM-tritonmoe/atom/model_ops/mellum_hf_gateup_profiles.json').read_text())['mi32_flags'],dtype=torch.int32)
for n in [1,990,2240]:
 test=ids if n==990 else torch.arange(n*8,device='cuda',dtype=torch.int32).reshape(n,8)%64
 header=torch.tensor([13,7],dtype=torch.int32,device='cuda');actual=ck_hf_korder_plan(test,header).cpu();counts=test.cpu().flatten().bincount(minlength=64)
 assert torch.equal(actual[:2],header.cpu()) and torch.equal(actual[2:],table[counts])
report['metadata_fixture_cases']=[1,990,2240]
with torch.inference_mode():
 first=mellum_ordered_moe(h,w13,w2,rw,ids,ActivationType.Silu)
 equal=True
 for _ in range(40):equal &= torch.equal(first,mellum_ordered_moe(h,w13,w2,rw,ids,ActivationType.Silu))
 report['fixed_production_routes_repeat40']=equal;assert equal
other=outroot/'fixed_gpu0.pt'
if tag=='gpu1' and other.exists():report['cross_gpu_equal']=torch.equal(first.cpu(),torch.load(other,weights_only=True));assert report['cross_gpu_equal']
torch.save(first.cpu(),outroot/f'fixed_{tag}.pt');mod=Path(os.environ['AITER_JIT_DIR'])/'module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so';report['binary_sha256']=hashlib.sha256(mod.read_bytes()).hexdigest();(outroot/f'fixed_validation_{tag}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
