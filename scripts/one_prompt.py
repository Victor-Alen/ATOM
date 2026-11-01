#!/usr/bin/env python3
"""Engine vs HF for one prompt, token by token."""
import json
import sys
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
pid = sys.argv[1] if len(sys.argv) > 1 else "A-0007"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == pid:
        rec = r
        break
fresh = json.load(open("/workspace/correctness/hf_scenario_a_20_fresh.json"))
ref = fresh[pid][:16]
payload = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]),
           "max_tokens": 16, "temperature": 0.0, "ignore_eos": True, "stream": False}
req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as resp:
    got = json.loads(resp.read())["token_ids"]
print(f"{pid} len={len(rec['prompt_token_ids'])}")
print("engine:", got)
print("hf    :", ref)
print("agree :", sum(1 for a, b in zip(got, ref) if a == b), "/", len(ref))
print("### ONE_PROMPT_DONE", flush=True)
