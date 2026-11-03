#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import struct

root=Path('/workspace/logs/step0_continuation')
records={key:json.loads((root/name).read_text()) for key,name in {
    'baseline':'up_fp64_validation_baseline_post.json',
    'hip0':'up_fp64_validation_gpu0.json',
    'hip1':'up_fp64_validation_gpu1.json',
    'service_smoke':'up_fp64_service_smoke/summary.json'}.items()}
for key in ('hip0','hip1'):
    assert records[key]['target_up_exact'] and records[key]['target_gated_exact']
    assert records[key]['repeats']==40 and records[key]['distinct_up_hashes']==1
    assert records[key]['gate_vs_baseline']['different']==0
assert records['hip0']['up_hash']==records['hip1']['up_hash']
def bf16_bits(value):
    return f'0x{struct.unpack("!I",struct.pack("!f",value))[0]>>16:04X}'
module=Path('/workspace/ck-mellum-up-fp64-jit/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so')
summary={'records':records,'ck_base':'d6e03ebd9','ck_fix':'14e5a4558',
    'up_bf16_bits':{key:bf16_bits(records[key]['up_bf16']) for key in ('baseline','hip0','hip1')},
    'hf_up_bf16_bits':bf16_bits(records['hip0']['hf_up_bf16']),
    'stage1_latency_ratio_hip0':records['hip0']['stage1_median_ms']/records['baseline']['stage1_median_ms'],
    'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest()}
(root/'up_fp64_fix_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({key:value for key,value in summary.items() if key!='records'},indent=2))
