import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
target = "C-0011"
rec = None
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == target:
        rec = r
        break
ref = rec["token_ids"][:64]
text = tok.decode(rec["prompt_token_ids"])
p = {"model": MODEL, "prompt": text, "max_tokens": 64, "temperature": 0.0,
     "ignore_eos": True, "stream": False}
rq = urllib.request.Request("http://127.0.0.1:8001/v1/completions", data=json.dumps(p).encode(),
                            headers={"Content-Type": "application/json"})
got = json.loads(urllib.request.urlopen(rq, timeout=7200).read())["token_ids"]
n = min(len(got), len(ref))
a = sum(1 for i in range(n) if got[i] == ref[i]) / n if n else 0
print(f"{target} alone on current server: agreement={a:.3f}")
print("got:", got[:8])
print("ref:", ref[:8])
