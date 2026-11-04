import argparse
import json
from pathlib import Path
import urllib.request
from transformers import AutoTokenizer
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root)
rec=next(r for r in map(json.loads,Path('/workspace/correctness/hf_scenario_a_20_eager.jsonl').read_text().splitlines()) if r['id']=='A-0002')
tok=AutoTokenizer.from_pretrained('/data/models/Mellum2.1-12B-A2.5B-Thinking')
assert tok.encode(rec['prompt'],add_special_tokens=False)==rec['prompt_token_ids']
payload={'model':'/data/models/Mellum2.1-12B-A2.5B-Thinking','prompt':rec['prompt'],'max_tokens':4,'temperature':0,'ignore_eos':True,'stream':False}
req=urllib.request.Request('http://127.0.0.1:8005/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=1200) as response:body=json.load(response)
(root/'request.json').write_text(json.dumps(body,indent=2))
print('PRODUCTION_TRACE_TOKENS',body['token_ids'],'HF',rec['token_ids'][:4],flush=True)
assert body['token_ids'][:3]==rec['token_ids'][:3]
assert (root/'atom/step_3.pt').exists()
