#!/usr/bin/env python3
"""Experiment B: teacher forcing. Feed prompt+ref[:k] and check whether the engine
predicts ref[k]. Reports the first step whose argmax differs from HF's reference."""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = None
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "C-0011":
        rec = r
        break
ref = rec["token_ids"]
base_ids = rec["prompt_token_ids"]
prefix_text = tok.decode(base_ids)

first_bad = None
for k in range(0, 10):
    text = prefix_text + tok.decode(ref[:k]) if k else prefix_text
    p = {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0.0,
         "ignore_eos": True, "stream": False}
    rq = urllib.request.Request("http://127.0.0.1:8001/v1/completions", data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    got = json.loads(urllib.request.urlopen(rq, timeout=7200).read())["token_ids"][0]
    want = ref[k]
    ok = got == want
    if not ok and first_bad is None:
        first_bad = k
    print(f"step {k}: engine={got} hf={want} {'OK' if ok else 'MISMATCH'}", flush=True)
print("first bad step:", first_bad, flush=True)
print("### TF_DONE", flush=True)
