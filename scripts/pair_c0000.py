#!/usr/bin/env python3
"""Which token does the engine append to the second request when the first output
token is not 233?"""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
rec = None
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "C-0000":
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
for i in range(2):
    req = urllib.request.Request(
        "http://127.0.0.1:8001/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        print(f"call{i}:", json.loads(resp.read())["token_ids"], flush=True)
print("prompt_len:", len(rec["prompt_token_ids"]), "last3:", rec["prompt_token_ids"][-3:], flush=True)
print("### PAIR2_DONE")
