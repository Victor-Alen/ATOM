#!/usr/bin/env python3
"""Score scenario A against freshly generated HF references."""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
fresh = json.load(open("/tmp/hf_a20_fresh.json"))
recs = [json.loads(l) for l in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8")]

matched = total = exact = 0
for rec in recs[:20]:
    pid = rec["id"]
    ref = fresh[pid][:64]
    payload = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]),
               "max_tokens": 64, "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        got = json.loads(resp.read())["token_ids"]
    n = min(len(got), len(ref))
    same = sum(1 for i in range(n) if got[i] == ref[i])
    matched += same
    total += n
    if same == n:
        exact += 1
    print(f"  {pid}: {same}/{n} len={len(rec['prompt_token_ids'])}", flush=True)
print(f"A(fresh): prompts=20 agree={matched/total:.4f} exact={exact}/20", flush=True)
print("### A_FRESH_SCORE_DONE", flush=True)
