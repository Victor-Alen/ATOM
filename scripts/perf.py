import json, time, urllib.request
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
p = {"model": MODEL, "prompt": text, "max_tokens": 64, "temperature": 0.0,
     "ignore_eos": True, "stream": False}
rq = urllib.request.Request("http://127.0.0.1:8003/v1/completions", data=json.dumps(p).encode(),
                            headers={"Content-Type": "application/json"})
t0 = time.time()
out = json.loads(urllib.request.urlopen(rq, timeout=3600).read())
dt = time.time() - t0
n = len(out["token_ids"])
print("tokens=%d  wall=%.3fs  tok/s=%.2f  ttft=%.3f  tpot=%.4f"
      % (n, dt, n / dt, out["usage"].get("ttft_s", -1), out["usage"].get("tpot_s", -1)))
