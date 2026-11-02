#!/usr/bin/env python3
"""Same prompt three times on a fresh server: are the answers identical?"""
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
           "max_tokens": 6, "temperature": 0.0, "ignore_eos": True, "stream": False}
for i in range(3):
    req = urllib.request.Request("http://127.0.0.1:8003/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        print(f"run{i}:", json.loads(resp.read())["token_ids"], flush=True)
print("### SAME3_DONE")
