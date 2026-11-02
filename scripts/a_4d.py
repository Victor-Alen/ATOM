import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
fresh = json.load(open("/workspace/correctness/hf_scenario_a_20_fresh.json"))
recs = [json.loads(l) for l in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8")]
tot, ex = 0.0, 0
for rec in recs:
    ref = fresh[rec["id"]][:64]
    text = tok.decode(rec["prompt_token_ids"])
    p = {"model": MODEL, "prompt": text, "max_tokens": 64, "temperature": 0.0,
         "ignore_eos": True, "stream": False}
    r = urllib.request.Request("http://127.0.0.1:8001/v1/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    got = json.loads(urllib.request.urlopen(r, timeout=3600).read())["token_ids"]
    n = min(len(got), len(ref))
    a = sum(1 for i in range(n) if got[i] == ref[i]) / n if n else 0.0
    tot += a
    ex += (a == 1.0)
    print(f"{rec['id']}: {a:.3f} {'EXACT' if a == 1.0 else ''}", flush=True)
print(f"A: agreement={tot/len(recs):.4f} exact={ex}/{len(recs)}", flush=True)
print("### A4D_DONE", flush=True)
