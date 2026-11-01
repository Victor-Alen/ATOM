#!/usr/bin/env python3
"""Scenario A (20 prompts) against freshly generated HF references."""
import json
import time
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
fresh = json.load(open("/workspace/correctness/hf_scenario_a_20_fresh_blas.json"))
recs = [json.loads(l) for l in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8")]

matched = total = exact = 0
t0 = time.time()
for rec in recs:
    ref = fresh[rec["id"]][:64]
    payload = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]),
               "max_tokens": 64, "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            got = json.loads(resp.read())["token_ids"]
    except Exception as exc:  # noqa: BLE001
        print(f"  {rec['id']}: ERR {type(exc).__name__}", flush=True)
        continue
    n = min(len(got), len(ref))
    same = sum(1 for i in range(n) if got[i] == ref[i])
    matched += same
    total += n
    exact += same == n
    print(f"  {rec['id']}: {same}/{n} len={len(rec['prompt_token_ids'])}", flush=True)
print(f"A(moedet): agree={matched/total:.4f} exact={exact}/{len(recs)} elapsed={time.time()-t0:.0f}s", flush=True)
print("### A_ONLY_DONE", flush=True)
