import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0000":
        rec = r
        break
text = tok.decode(rec["prompt_token_ids"])
outs = []
for i in range(3):
    p = {"model": MODEL, "prompt": text, "max_tokens": 16, "temperature": 0.0,
         "ignore_eos": True, "stream": False}
    rq = urllib.request.Request("http://127.0.0.1:8003/v1/completions", data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    outs.append(json.loads(urllib.request.urlopen(rq, timeout=1800).read())["token_ids"])
print("run0:", outs[0])
print("run1:", outs[1])
print("run2:", outs[2])
print("identical:", outs[0] == outs[1] == outs[2])
