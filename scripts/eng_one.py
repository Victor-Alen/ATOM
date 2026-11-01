#!/usr/bin/env python3
"""Engine greedy tokens for A-0006 (first 8)."""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0006":
        rec = r
        break
payload = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]),
           "max_tokens": 8, "temperature": 0.0, "ignore_eos": True, "stream": False}
req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as resp:
    got = json.loads(resp.read())["token_ids"]
print("engine A-0006:", got, flush=True)
print("hf_now A-0006:", json.load(open("/tmp/hf_a0006.json")), flush=True)
print("### ENG_ONE_DONE", flush=True)
