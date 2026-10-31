#!/usr/bin/env python3
"""Is A-0002 deterministic? Run it twice and compare."""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0002":
        rec = r
        break
text = tok.decode(rec["prompt_token_ids"])


def gen(steps=8):
    payload = {"model": MODEL, "prompt": text, "max_tokens": steps,
               "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())["token_ids"]


a = gen()
b = gen()
print("run1:", a, flush=True)
print("run2:", b, flush=True)
print("deterministic:", a == b, flush=True)
print("hf      :", rec["token_ids"][:8], flush=True)
print("### DET_DONE", flush=True)
