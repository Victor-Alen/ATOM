import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
STEPS = 24
rows = []
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if len(r["prompt_token_ids"]) <= 12000:
        rows.append(r)
    if len(rows) >= 12:
        break
tot, ex = 0.0, 0
for rec in rows:
    text = tok.decode(rec["prompt_token_ids"])
    p = {"model": MODEL, "prompt": text, "max_tokens": STEPS, "temperature": 0.0,
         "ignore_eos": True, "stream": False}
    rq = urllib.request.Request("http://127.0.0.1:8001/v1/completions", data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    got = json.loads(urllib.request.urlopen(rq, timeout=7200).read())["token_ids"]
    ref = rec["token_ids"][:STEPS]
    n = min(len(got), len(ref))
    a = sum(1 for i in range(n) if got[i] == ref[i]) / n if n else 0.0
    tot += a
    ex += (a == 1.0)
    print(f"{rec['id']}: {a:.3f} {'EXACT' if a == 1.0 else ''}", flush=True)
print(f"C clean: agreement={tot/len(rows):.4f} exact={ex}/{len(rows)}", flush=True)
print("### CCLEAN_DONE", flush=True)
