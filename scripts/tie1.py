import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
recs = {}
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] in ("A-0003", "A-0000"):
        recs[r["id"]] = r
HF_TOP1 = {"A-0003": (233, 0.125), "A-0000": (233, 2.0)}
for pid in ("A-0003", "A-0000"):
    rec = recs[pid]
    p = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]), "max_tokens": 1,
         "temperature": 0.0, "ignore_eos": True, "stream": False}
    rq = urllib.request.Request("http://127.0.0.1:8003/v1/completions", data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    got = json.loads(urllib.request.urlopen(rq, timeout=3600).read())["token_ids"]
    hf_top1, margin = HF_TOP1[pid]
    print(f"{pid}: HF top1={hf_top1} margin={margin}  production={got}  "
          f"{'SAME' if got and got[0] == hf_top1 else 'FLIPPED'}", flush=True)
print("### TIE1_DONE", flush=True)
