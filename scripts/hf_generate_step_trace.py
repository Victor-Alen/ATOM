"""Capture an unmodified official HF generate step, with reference verification."""
import os,json,sys,importlib.util,time,torch
from pathlib import Path
from hf_reference_generate import MODEL_PATH,load_mellum_classes,load_checkpoint_experts
root=Path(os.environ['MELLUM_TRACE_ROOT']);root.mkdir(parents=True,exist_ok=True);target=int(os.environ['MELLUM_TRACE_STEP']);torch.set_num_threads(8)
rec=json.loads(Path('/workspace/correctness/hf_scenario_a_20_eager.jsonl').read_text().splitlines()[0]);Config,LM=load_mellum_classes();mod=sys.modules['transformers.models.mellum.modeling_mellum'];cfg=Config.from_pretrained(MODEL_PATH)
base=mod.MellumPreTrainedModel;original=base._init_weights;skipped=[]
def init(self,module):
 if isinstance(module,mod.MellumExperts):skipped.append(1);return
 return original(self,module)
base._init_weights=init;start=time.monotonic()
try:model,info=LM.from_pretrained(MODEL_PATH,config=cfg,dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True)
finally:base._init_weights=original
expected={f'model.layers.{i}.mlp.experts.{n}' for i in range(28) for n in ['gate_up_proj','down_proj']};assert set(info['missing_keys'])==expected;assert not info['mismatched_keys']
load_checkpoint_experts(model,MODEL_PATH);model=model.cuda().eval();print('HF_READY',time.monotonic()-start,'skipped',len(skipped),flush=True)
spec=importlib.util.spec_from_file_location('standalone_step_trace','/workspace/ATOM-tritonmoe/atom/models/step_tensor_trace.py');trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
recorder=trace.TensorRecorder(root/'hf',target);recorder.attach(model.model);counter=[0]
def begin(module,args,kwargs):
 step=counter[0];counter[0]+=1;recorder.begin(step)
 ids=kwargs.get('input_ids',args[0] if args else None);pos=kwargs.get('position_ids')
 recorder.add('model.input_ids',ids)
 if pos is not None:recorder.add('model.positions',pos.reshape(-1))
 recorder.data['model.generate_step']=step if recorder.active else None
 if recorder.active:
  recorder.data['model.forward_kwargs']={k:(list(v.shape) if torch.is_tensor(v) else type(v).__name__) for k,v in kwargs.items()}
def head(module,args,out):
 if recorder.active:recorder.add('head.logits',out);recorder.save()
model.model.register_forward_pre_hook(begin,with_kwargs=True);model.lm_head.register_forward_hook(head)
prompt=torch.tensor([rec['prompt_token_ids']],device='cuda');
with torch.inference_mode():out=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),max_new_tokens=64,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
pred=out[0,prompt.shape[1]:].tolist();same=pred==rec['token_ids'][:64];(root/'hf_generate.json').write_text(json.dumps({'token_ids':pred,'historical_reference_matches':same,'target':target,'skipped_expert_initializations':len(skipped)},indent=2)+'\n');print('REFERENCE_MATCH',same,flush=True);assert same
