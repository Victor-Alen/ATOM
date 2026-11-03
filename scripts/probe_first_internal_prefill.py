import json,torch
from pathlib import Path
from safetensors import safe_open
root=Path('/workspace/logs/step0_continuation/hf_attn_first29');a=torch.load(root/'atom/step_29.pt',weights_only=False,map_location='cpu')['tensors']
model=Path('/data/models/Mellum2.1-12B-A2.5B-Thinking');wm=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
def w(n):
 with safe_open(model/wm[n],framework='pt',device='cpu') as f:return f.get_tensor(n)
rec=json.loads(Path('/workspace/correctness/hf_scenario_a_20_eager.jsonl').read_text().splitlines()[0]);x=w('model.embed_tokens.weight')[rec['prompt_token_ids']].cuda();weight=w('model.layers.0.input_layernorm.weight').cuda()
f=x.float();ref=weight*(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-6)).bfloat16();actual=a['L0.step0.qkv_proj.in0'].cuda();diff=ref!=actual;coords=diff.nonzero();rows=[]
for token,feature in coords[:20].tolist():rows.append({'token':token,'feature':feature,'hf':float(ref[token,feature]),'actual':float(actual[token,feature]),'input':float(x[token,feature])})
r={'norm_prefill_different':int(diff.sum()),'total':diff.numel(),'first20':rows,'max':float((ref.float()-actual.float()).abs().max())}
(root/'first_internal_prefill.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
