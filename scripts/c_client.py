#!/usr/bin/env python3
"""Send a C prompt prefix to the engine so its layer-0 dumps are produced."""
import json
import sys
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
target = sys.argv[1] if len(sys.argv) > 1 else "C-0002"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1200

rec = None
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == target:
        rec = r
        break
tok = AutoTokenizer.from_pretrained(MODEL)
text = tok.decode(rec["prompt_token_ids"][:n])
p = {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0.0,
     "ignore_eos": True, "stream": False}
r = urllib.request.Request("http://127.0.0.1:8001/v1/completions", data=json.dumps(p).encode(),
                           headers={"Content-Type": "application/json"})
out = json.loads(urllib.request.urlopen(r, timeout=3600).read())
print("first token:", out["token_ids"], flush=True)
json.dump(text, open("/tmp/c1200_text.json", "w"))
print("### CCLIENT_DONE", flush=True)
