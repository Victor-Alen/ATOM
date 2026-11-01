#!/usr/bin/env python3
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0012":
        rec = r
        break
payload = {
    "model": MODEL,
    "prompt": tok.decode(rec["prompt_token_ids"]),
    "max_tokens": 3,
    "temperature": 0.0,
    "ignore_eos": True,
    "stream": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=800) as r:
    got = json.loads(r.read())["token_ids"]
json.dump(
    {"id": "A-0012", "prompt_token_ids": rec["prompt_token_ids"], "engine_tokens": got},
    open("/tmp/a0012_run.json", "w"),
)
print("engine tokens:", got, flush=True)
