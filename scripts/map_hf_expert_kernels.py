import torch,json
from pathlib import Path
r=Path('/workspace/logs/step0_continuation/norm990_active_prod');p=json.loads((r/'hf_grouped_gateup_profile_summary.json').read_text());hf=torch.load('/workspace/logs/step0_continuation/official_generate_prefill_full/hf/step_0.pt',weights_only=False)['tensors'];ids=hf['L0.full.router.out2'].flatten();counts=ids.bincount(minlength=64);active=(counts>0).nonzero().flatten().tolist();assert len(active)==len(p['kernels'])
for e,k in zip(active,p['kernels']):
 if e in [55,49,23,7]:print('EXPERT',e,'M',int(counts[e]),'KERNEL',k['name'],'LAUNCH',k['launch'])
print('FIRST',p['first_gate_examples'][0])
