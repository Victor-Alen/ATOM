"""Fresh native HF greedy A0 reference; independent cache per repeat."""
import json,os,time
from pathlib import Path
import torch
from transformers.initialization import no_init_weights
from hf_reference_generate import MODEL_PATH,load_mellum_classes,load_checkpoint_experts
root=Path('/workspace/logs/step0_continuation/hf_current64');root.mkdir(exist_ok=True)
torch.set_num_threads(8)
rec=json.loads(Path('/workspace/correctness/hf_scenario_a_20_eager.jsonl').read_text().splitlines()[0])
Config,LM=load_mellum_classes();cfg=Config.from_pretrained(MODEL_PATH)
# Every missing packed expert parameter is overwritten by the checkpoint loader.
with no_init_weights():
 model=LM.from_pretrained(MODEL_PATH,config=cfg,dtype=torch.bfloat16,attn_implementation='eager')
load_checkpoint_experts(model,MODEL_PATH);model=model.cuda().eval();print('HF_READY',flush=True)
prompt=torch.tensor([rec['prompt_token_ids']],device='cuda',dtype=torch.long);runs=[]
with torch.inference_mode():
 for run in range(2):
  cache=None;pred=[]
  for step in range(64):
   ids=prompt if step==0 else torch.tensor([[pred[-1]]],device='cuda',dtype=torch.long)
   mask=torch.ones((1,prompt.shape[1]+step),device='cuda',dtype=torch.long)
   out=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
   cache=out.past_key_values;pred.append(int(out.logits[0,-1].argmax()))
  runs.append(pred);print('RUN',run,pred,flush=True)
assert runs[0]==runs[1], 'HF repeats differ'
ref={**rec,'token_ids':runs[0]};(root/'hf_a0_64.jsonl').write_text(json.dumps(ref,ensure_ascii=False)+'\n')
(root/'repeat.json').write_text(json.dumps({'runs':runs,'same':True,'historical_first_mismatch':next((i for i,(a,b) in enumerate(zip(runs[0],rec['token_ids'])) if a!=b),None)},indent=2)+'\n')
