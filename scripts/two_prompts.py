#!/usr/bin/env python3
"""Send A-0006 and A-0002 in one session to see whether the engine distinguishes them."""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
recs = {}
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    recs[r["id"]] = r


def run(pid, steps=6):
    payload = {"model": MODEL, "prompt": tok.decode(recs[pid]["prompt_token_ids"]),
               "max_tokens": steps, "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8003/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())["token_ids"]


for pid in ("A-0006", "A-0002", "A-0006"):
    try:
        print(f"{pid}:", run(pid), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{pid}: ERR {type(exc).__name__}", flush=True)
print("### TWO_PROMPTS_DONE")
