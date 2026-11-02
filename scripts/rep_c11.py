import json, urllib.request
from transformers import AutoTokenizer
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = None
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "C-0011":
        rec = r
        break
text = tok.decode(rec["prompt_token_ids"])
outs = []
for k in range(3):
    p = {"model": MODEL, "prompt": text, "max_tokens": 24, "temperature": 0.0,
         "ignore_eos": True, "stream": False}
    rq = urllib.request.Request("http://127.0.0.1:8001/v1/completions", data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    got = json.loads(urllib.request.urlopen(rq, timeout=7200).read())["token_ids"]
    outs.append(got)
    print(f"run{k}: {got[:12]}", flush=True)
print("all three identical:", outs[0] == outs[1] == outs[2], flush=True)
ref = rec["token_ids"][:24]
n = min(len(outs[0]), len(ref))
print("agreement run0 vs hf:", round(sum(1 for i in range(n) if outs[0][i] == ref[i]) / n, 3), flush=True)
print("hf ref:", ref[:12], flush=True)
print("### REP_DONE", flush=True)
