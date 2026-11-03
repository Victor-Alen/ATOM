import json,time,urllib.request,sys
from pathlib import Path
out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
refs=[json.loads(x) for x in Path('/workspace/correctness/hf_scenario_a_20_eager.jsonl').read_text().splitlines()][:2];rows=[]
for ref in refs:
 req=urllib.request.Request('http://127.0.0.1:8003/v1/completions',data=json.dumps({'model':'/data/models/Mellum2.1-12B-A2.5B-Thinking','prompt':ref['prompt'],'max_tokens':64,'temperature':0.,'ignore_eos':True,'stream':False}).encode(),headers={'Content-Type':'application/json'})
 start=time.perf_counter()
 with urllib.request.urlopen(req,timeout=900) as f:response=json.load(f)
 elapsed=time.perf_counter()-start;ids=response['token_ids'];gold=ref['token_ids'][:64];assert len(ids)==64
 row={'id':ref['id'],'prompt_tokens':ref['prompt_tokens'],'elapsed_seconds':elapsed,'end_to_end_output_tokens_per_second':64/elapsed,'matches':sum(a==b for a,b in zip(ids,gold)),'total':64,'first_mismatch':next((i for i,(a,b) in enumerate(zip(ids,gold)) if a!=b),None),'token_ids':ids};rows.append(row);out.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
