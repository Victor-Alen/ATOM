import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = [json.loads(l) for l in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8") if json.loads(l)["id"] == "A-0000"][0]
p = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]), "max_tokens": 16,
     "temperature": 0.0, "ignore_eos": True, "stream": False}
rq = urllib.request.Request("http://127.0.0.1:8003/v1/completions", data=json.dumps(p).encode(),
                            headers={"Content-Type": "application/json"})
print("TOKENS:", json.loads(urllib.request.urlopen(rq, timeout=1800).read())["token_ids"])
