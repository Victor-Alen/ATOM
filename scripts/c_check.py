import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
recs = {}
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] in ("C-0002", "C-0001", "C-0003"):
        recs[r["id"]] = r
HF_FRESH = {
    "C-0001": [233, 233, 37, 6539, 233, 910, 1132, 97],
}
for pid in ("C-0002", "C-0003", "C-0001"):
    rec = recs[pid]
    p = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]), "max_tokens": 8,
         "temperature": 0.0, "ignore_eos": True, "stream": False}
    rq = urllib.request.Request("http://127.0.0.1:8003/v1/completions", data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    got = json.loads(urllib.request.urlopen(rq, timeout=3600).read())["token_ids"]
    want = HF_FRESH.get(pid, rec["token_ids"][:8])
    m = sum(1 for a, b in zip(got, want) if a == b)
    print(f"{pid} (len={len(rec['prompt_token_ids'])}): {m}/8  eng={got[:6]} want={want[:6]}", flush=True)
print("### CCHECK_DONE", flush=True)
