import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
fresh = json.load(open("/workspace/correctness/hf_scenario_a_20_fresh.json"))
recs = [json.loads(l) for l in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8")][:6]
tot = 0.0
for rec in recs:
    ref = fresh[rec["id"]][:64]
    p = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]), "max_tokens": 64,
         "temperature": 0.0, "ignore_eos": True, "stream": False}
    rq = urllib.request.Request("http://127.0.0.1:8002/v1/completions", data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    got = json.loads(urllib.request.urlopen(rq, timeout=3600).read())["token_ids"]
    n = min(len(got), len(ref))
    a = sum(1 for i in range(n) if got[i] == ref[i]) / n if n else 0
    tot += a
    print(f"{rec['id']}: {a:.3f}", flush=True)
print(f"FAST PATH A(6): agreement={tot/len(recs):.4f}", flush=True)
print("### FAST_A_DONE", flush=True)
