import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import torch
import torch.nn.functional as F
from safetensors import safe_open
from aiter.ops.shuffle import shuffle_weight
from transformers.integrations.moe import grouped_mm_experts_forward,batched_mm_experts_forward
from atom.model_ops import mellum_hf_gemm_moe as backend

root=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking')
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
with ExitStack() as stack:
 handles={}
 def read(name):
  shard=index[name]
  if shard not in handles:handles[shard]=stack.enter_context(safe_open(str(root/shard),framework='pt',device='cpu'))
  return handles[shard].get_tensor(name)
 w13=torch.stack([torch.cat([read(f'model.layers.0.mlp.experts.{e}.gate_proj.weight'),read(f'model.layers.0.mlp.experts.{e}.up_proj.weight')]) for e in range(64)]).cuda()
 w2=torch.stack([read(f'model.layers.0.mlp.experts.{e}.down_proj.weight') for e in range(64)]).cuda()
cap=Path('/workspace/logs/step0_causal_boundary/A-0003/capture_isolated_0')
x=torch.load(cap/'mlp_in0.pt',map_location='cpu',weights_only=True).cuda()
router=torch.load(cap/'router_out0.pt',map_location='cpu',weights_only=True).cuda()
def gate(t):
 g,u=t.chunk(2,dim=-1);return F.silu(g)*u
ref_module=SimpleNamespace(gate_up_proj=w13,down_proj=w2,num_experts=64,has_bias=False,is_transposed=False,_apply_gate=gate)
layer=SimpleNamespace(w13_weight=shuffle_weight(w13),w2_weight=shuffle_weight(w2),tp_size=1,dp_size=1,ep_size=1,top_k=8,renormalize=True)
reports=[]
for count in [x.shape[0],1]:
 a=x[:count];r=router[:count]
 weights,ids=F.softmax(r,dtype=torch.float32,dim=-1).topk(8,-1)
 weights=(weights/weights.sum(-1,keepdim=True)).to(r.dtype)
 function=grouped_mm_experts_forward if count>1 else batched_mm_experts_forward
 reference=function(ref_module,a,ids,weights)
 backend.get_forward_context=lambda:SimpleNamespace(context=SimpleNamespace(is_prefill=count>1))
 actual=backend.apply(layer,a,r)
 d=(actual.float()-reference.float()).abs()
 report={'rows':count,'different':int((d!=0).sum()),'max':float(d.max()),'total':d.numel()}
 reports.append(report);print(json.dumps(report),flush=True)
assert not any(r['different'] for r in reports)
Path('/workspace/logs/step0_causal_boundary/hf_gemm_moe_probe.json').write_text(json.dumps(reports,indent=2))
print('HF_GEMM_MOE_PROBE_EXACT',flush=True)
