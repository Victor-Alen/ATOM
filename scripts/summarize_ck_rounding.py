#!/usr/bin/env python3
"""Collect the frozen CK rounding probes and end-to-end regression summaries."""
import hashlib
import json
from pathlib import Path

root=Path('/workspace/logs/step0_continuation')
def read(name):
    return json.loads((root/name).read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

out={"date":"2026-11-03", "probes":{}, "service":{}, "sha256":{}}
for name in ('ck_round_prefill866.json','ck_round_decode1.json',
             'ck_stage12_round_prefill866.json','ck_stage12_round_decode1.json',
             'ck_round_fp64_prefill866.json'):
    out['probes'][name]=read(name)
for variant in ('control','stage1','stage12'):
    out['service'][variant]={}
    for scenario in 'abc':
        if variant == 'control':
            name={'a':'ordered_service_a20_64','b':'ordered_service_b20_64','c':'ck_round_control_c20'}[scenario]
        else:
            name=f'ck_{variant}_round_{scenario}20'
        out['service'][variant][scenario]=read(name+'/summary.json')
for path in ('/workspace/correctness/hf_scenario_a_20_eager.jsonl',
             '/workspace/correctness/hf_scenario_b_20_eager.jsonl',
             str(root/'ck_round_fresh_hf_c20.jsonl'),
             '/workspace/ck-mellum-hf-jit/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so',
             '/workspace/ck-mellum-hf-stage12-jit/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so'):
    out['sha256'][path]=sha(path)
fresh=[json.loads(s) for s in (root/'ck_round_fresh_hf_c20.jsonl').read_text().splitlines()]
saved={r['id']:r for r in map(json.loads,Path('/workspace/correctness/hf_scenario_c_200_eager.jsonl').read_text().splitlines())}
out['fresh_c_vs_saved']={'prompts':len(fresh),'equal_prefixes':sum(r['token_ids']==saved[r['id']]['token_ids'][:8] for r in fresh)}
(root/'ck_hf_rounding_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['service'],indent=2))
