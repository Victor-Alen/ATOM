import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch
import transformers

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--pid',type=int,required=True);args=ap.parse_args()
root=Path(args.root);repo=Path('/workspace/ATOM-general-norm')
environment=dict(item.split('=',1) for item in Path(f'/proc/{args.pid}/environ').read_bytes().decode().split('\0') if '=' in item)
flags={k:v for k,v in environment.items() if k.startswith(('ATOM_MELLUM','ATOM_ENABLE_QK','ATOM_REFERENCE','ATOM_CAUSAL','ATOM_STEP_TRACE','ATOM_LAYER_TRACE','HIP_VISIBLE','ROCR_VISIBLE','CK_','AITER_JIT','AITER_META'))}
assert environment.get('ATOM_REFERENCE_MODE')!='1'
assert not any(environment.get(k) for k in ['ATOM_CAUSAL_DIR','ATOM_STEP_TRACE_DIR','ATOM_LAYER_TRACE_DIR'])
for k in ['ATOM_MELLUM_EXACT_NORM','ATOM_MELLUM_EXACT_LINEAR','ATOM_MELLUM_EXACT_ROPE','ATOM_MELLUM_BF16_HF_ATTENTION','ATOM_MELLUM_HF_GEMM_MOE']:
 assert environment.get(k)=='1',k
props=torch.cuda.get_device_properties(0)
assert props.pci_bus_id==34,props
files=[repo/'atom/model_ops'/name for name in ['mellum_bf16_attention.py','mellum_hf_rope.py','mellum_hf_gemm_moe.py','mellum_hf_norm.py','layernorm.py','linear.py','attention_mha.py','moe.py','embed_head.py']]
files+=[Path('/workspace/correctness')/name for name in ['hf_scenario_a_20_eager.jsonl','hf_scenario_b_20_eager.jsonl','hf_scenario_c_200_eager.jsonl']]
sha={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
report={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        'server_pid':args.pid,'server_cmdline':Path(f'/proc/{args.pid}/cmdline').read_bytes().decode().replace('\0',' '),
        'server_cwd':os.readlink(f'/proc/{args.pid}/cwd'),'flags':flags,'torch':torch.__version__,
        'transformers':transformers.__version__,'device':str(props),'pci':'0000:22:00.0','sha256':sha,
        'scope':{'A':20,'B':20,'C':200,'output_tokens':64,'threshold':0.9,'single_client':True},
        'backend':'production ATOM model and paged KV; explicit HF-order GPU operators; grouped/batched experts'}
(root/'runtime_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['sha256','flags']},indent=2),flush=True)
