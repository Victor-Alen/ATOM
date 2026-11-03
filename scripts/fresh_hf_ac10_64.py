"""Fresh official HF generate references for the first 10 A and C requests."""
import sys,os,time,json,hashlib,torch
from pathlib import Path
sys.path.insert(0,'/workspace/ATOM-tritonmoe/scripts')
from hf_reference_generate import MODEL_PATH,load_mellum_classes,load_checkpoint_experts
from transformers import AutoTokenizer
outroot=Path('/workspace/logs/step0_continuation/ck_hf_profile_prod/fresh_hf');outroot.mkdir(exist_ok=True);torch.set_num_threads(8)
Config,LM=load_mellum_classes();mod=sys.modules['transformers.models.mellum.modeling_mellum'];cfg=Config.from_pretrained(MODEL_PATH);base=mod.MellumPreTrainedModel;original=base._init_weights;skipped=[]
def init(self,module):
 if isinstance(module,mod.MellumExperts):skipped.append(1);return
 return original(self,module)
base._init_weights=init
try:model,info=LM.from_pretrained(MODEL_PATH,config=cfg,dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True)
finally:base._init_weights=original
expected={f'model.layers.{i}.mlp.experts.{n}' for i in range(28) for n in ['gate_up_proj','down_proj']};assert set(info['missing_keys'])==expected and not info['mismatched_keys'];assert len(skipped)==28
load_checkpoint_experts(model,MODEL_PATH);model=model.cuda().eval();tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH);print('HF_READY',flush=True)
a=[json.loads(x) for x in Path('/workspace/correctness/hf_scenario_a_20_eager.jsonl').read_text().splitlines()][:10];c=[json.loads(x) for x in Path('/data/datasets/scenario_c_prompts.jsonl').read_text().splitlines()][:10]
def generate(ids):
 prompt=torch.tensor([ids],device='cuda')
 with torch.inference_mode():out=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),max_new_tokens=64,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)
 pred=out[0,len(ids):].tolist();assert len(pred)==64;return pred
outputs={}
for scenario,records in [('A',a),('C',c)]:
 rows=[];path=outroot/f'{scenario}10_64.jsonl'
 with path.open('w') as f:
  for rec in records:
   ids=tokenizer.encode(rec['prompt'],add_special_tokens=False)
   if scenario=='A':assert ids==rec['prompt_token_ids']
   start=time.perf_counter();pred=generate(ids);elapsed=time.perf_counter()-start
   row={'id':rec['id'],'prompt':rec['prompt'],'prompt_token_ids':ids,'prompt_tokens':len(ids),'token_ids':pred,'elapsed_s':elapsed,'attention':'eager','dtype':'bfloat16','protocol':'official model.generate max_new_tokens64 do_sampleFalse use_cacheTrue eos_token_idNone pad_token_id0'}
   if scenario=='A':row['historical_reference_matches']=pred==rec['token_ids'][:64]
   f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();rows.append(row);print(scenario,rec['id'],'prompt_tokens',len(ids),'elapsed',round(elapsed,3),'historical_match',row.get('historical_reference_matches'),flush=True)
 outputs[scenario]=rows;print(scenario+'_DONE',len(rows),flush=True)
checks=[]
for rec in outputs['C'][:2]:
 pred=generate(rec['prompt_token_ids']);equal=pred==rec['token_ids'];checks.append({'id':rec['id'],'repeat_same_process_equal':equal});assert equal,checks
manifest={'protocol':'official HF generate; freshly tokenized dataset inputs; no cached C output used','max_new_tokens':64,'A_historical_reference_matches':all(r['historical_reference_matches'] for r in outputs['A']),'C_repeat_checks':checks,'sha256':{s:hashlib.sha256((outroot/f'{s}10_64.jsonl').read_bytes()).hexdigest() for s in ['A','C']},'transformers':__import__('transformers').__version__,'torch':torch.__version__}
(outroot/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('ALL_DONE',json.dumps(manifest),flush=True)
