import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,'/workspace/ATOM-general-norm/scripts')
from hf_reference_generate import MODEL_PATH,load_mellum_classes,load_checkpoint_experts
from atom.models.step_tensor_trace import TensorRecorder

ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--step',type=int,required=True);ap.add_argument('--root',required=True)
args=ap.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
rec=next(r for r in map(json.loads,Path('/workspace/correctness/hf_scenario_a_20_eager.jsonl').read_text().splitlines()) if r['id']==args.id)
Config,LM=load_mellum_classes();module=sys.modules['transformers.models.mellum.modeling_mellum']
original_init=module.MellumPreTrainedModel._init_weights
def init(self,child):
 if not isinstance(child,module.MellumExperts):return original_init(self,child)
module.MellumPreTrainedModel._init_weights=init
try:model,info=LM.from_pretrained(MODEL_PATH,config=Config.from_pretrained(MODEL_PATH),dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True)
finally:module.MellumPreTrainedModel._init_weights=original_init
assert set(info['missing_keys'])=={f'model.layers.{i}.mlp.experts.{n}' for i in range(28) for n in ('gate_up_proj','down_proj')}
assert not info['mismatched_keys']
load_checkpoint_experts(model,MODEL_PATH);model=model.cuda().eval()
recorder=TensorRecorder(root/'hf',args.step);recorder.attach(model.model);counter=[-1]
def begin(module,positional,kwargs):
 counter[0]+=1;recorder.begin(counter[0]);recorder.add('model.input_ids',kwargs.get('input_ids',positional[0] if positional else None))
 recorder.add('model.positions',kwargs.get('position_ids'))
def head(module,positional,out):
 if recorder.active:recorder.add('head.logits',out);recorder.save()
model.model.register_forward_pre_hook(begin,with_kwargs=True);model.lm_head.register_forward_hook(head)
for name in ['q_proj','k_proj','v_proj','q_norm','k_norm','o_proj']:
 sub=getattr(model.model.layers[0].self_attn,name)
 def before(module,positional,kwargs,name=name):
  if counter[0]>args.step:return
  for key,value in list(enumerate(positional))+list(kwargs.items()):
   if torch.is_tensor(value):recorder.detail[f'L0.step{counter[0]}.{name}.in{key}']=value.detach().cpu().clone()
 def after(module,positional,out,name=name):
  if counter[0]>args.step:return
  if torch.is_tensor(out):recorder.detail[f'L0.step{counter[0]}.{name}.out0']=out.detach().cpu().clone()
 sub.register_forward_pre_hook(before,with_kwargs=True);sub.register_forward_hook(after)
ids=torch.tensor([rec['prompt_token_ids']],device='cuda')
with torch.inference_mode():out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=args.step+1,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
got=out[0,ids.shape[1]:].tolist();expected=rec['token_ids'][:args.step+1]
assert got==expected,(got,expected)
(root/'hf_verified.json').write_text(json.dumps({'id':args.id,'step':args.step,'got':got,'expected':expected,'matched':True},indent=2))
print('OFFICIAL_HF_TRACE_VERIFIED',args.id,got,flush=True)
