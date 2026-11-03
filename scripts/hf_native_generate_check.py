"""Validate native generate against the fresh cached greedy protocol."""
import json,time,torch,sys
from pathlib import Path
from hf_reference_generate import MODEL_PATH,load_mellum_classes,load_checkpoint_experts
root=Path('/workspace/logs/step0_continuation/hf_current64');torch.set_num_threads(8)
Config,LM=load_mellum_classes();cfg=Config.from_pretrained(MODEL_PATH);mod=sys.modules['transformers.models.mellum.modeling_mellum'];original=LM._init_weights;skipped=[]
def init(self,module):
 if isinstance(module,mod.MellumExperts):skipped.append(1);return
 return original(self,module)
LM._init_weights=init;start=time.monotonic()
try:
 model,info=LM.from_pretrained(MODEL_PATH,config=cfg,dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True)
finally:LM._init_weights=original
expected={f'model.layers.{i}.mlp.experts.{n}' for i in range(28) for n in ['gate_up_proj','down_proj']}
assert set(info['missing_keys'])==expected,info['missing_keys']
assert not info['mismatched_keys'],info['mismatched_keys']
load_checkpoint_experts(model,MODEL_PATH);model=model.cuda().eval();print('HF_READY',time.monotonic()-start,'skipped_experts',len(skipped),flush=True)
rec=json.loads((root/'hf_a0_64.jsonl').read_text());p=torch.tensor([rec['prompt_token_ids']],device='cuda');mask=torch.ones_like(p)
with torch.inference_mode():
 out=model.generate(input_ids=p,attention_mask=mask,max_new_tokens=64,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
pred=out[0,p.shape[1]:].tolist();r={'tokens':pred,'same_as_cached_greedy':pred==rec['token_ids'],'skipped_expert_initializations':len(skipped),'all_missing_keys_were_loaded_experts':True,'load_seconds':time.monotonic()-start}
(root/'native_generate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
