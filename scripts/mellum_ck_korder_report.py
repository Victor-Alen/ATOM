from pathlib import Path
import json,torch,hashlib
r=Path('/workspace/logs/step0_continuation/norm990_active_prod');b=torch.load(r/'ck_stage_isolation.pt',weights_only=False);c=torch.load(r/'ck_stage_isolation_korder.pt',weights_only=False);report={};ids=b['ids']
for e in [55,6,49]:
 mask=ids==e;report[str(e)]={n:int((c[n+'_ck'][mask]!=b[n+'_hf'][mask]).sum()) for n in ['gate','up']}
report['first_scalar']={'ck':float(c['gate_ck'][0,0,407]),'hf':float(b['gate_hf'][0,0,407])}
report['new_first_gate']=json.loads((r/'ck_stage_isolation_korder.json').read_text())['first_gate'][0];report['new_first_up']=json.loads((r/'ck_stage_isolation_korder.json').read_text())['first_up'][0];report['new_first_gated']=json.loads((r/'ck_stage_isolation_korder.json').read_text())['first_gated'][0]
active=ids.flatten().bincount(minlength=64).nonzero().flatten().tolist();profiles=json.loads((r/'hf_grouped_gateup_profile_summary.json').read_text())['kernels'];report['expert6_kernel']=profiles[active.index(6)]['name'];report['expert6_m']=int((ids==6).sum());report['sha256']=hashlib.sha256(Path('/workspace/ck-mellum-hf-korder-jit/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so').read_bytes()).hexdigest();(r/'ck_korder_first_difference.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
