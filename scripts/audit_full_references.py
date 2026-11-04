import json
from pathlib import Path
from transformers import AutoTokenizer

base=Path('/workspace/correctness')
fresh=Path('/workspace/logs/step0_continuation/ck_hf_profile_prod/fresh_hf')
tokenizer=AutoTokenizer.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking')
def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
a=read(base/'hf_scenario_a_512_eager.jsonl');c=read(base/'hf_scenario_c_200_eager.jsonl')
a20=read(base/'hf_scenario_a_20_eager.jsonl')
report={}
for name,rows,small in [('A512',a,read(fresh/'A10_64.jsonl')),('C200',c,read(fresh/'C10_64.jsonl'))]:
 lookup={r['id']:r for r in rows}
 mismatch=[r['id'] for r in rows if tokenizer.encode(r['prompt'],add_special_tokens=False)!=r['prompt_token_ids']]
 overlap=[r['id'] for r in small if r['id'] not in lookup or r['prompt_token_ids']!=lookup[r['id']]['prompt_token_ids'] or r['token_ids'][:64]!=lookup[r['id']]['token_ids'][:64]]
 report[name]={'count':len(rows),'unique_ids':len(lookup),'short_outputs':[r['id'] for r in rows if len(r['token_ids'])<64],
               'input_tokenizer_mismatch':mismatch,'fresh10_mismatch':overlap,
               'min_prompt':min(len(r['prompt_token_ids']) for r in rows),'max_prompt':max(len(r['prompt_token_ids']) for r in rows)}
lookup={r['id']:r for r in a}
report['A20_vs_A512_mismatch']=[r['id'] for r in a20 if r['prompt_token_ids']!=lookup[r['id']]['prompt_token_ids'] or r['token_ids'][:64]!=lookup[r['id']]['token_ids'][:64]]
path=Path('/workspace/logs/step0_causal_boundary/full_reference_audit.json')
path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
