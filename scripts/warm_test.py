#!/usr/bin/env python3
"""Warm the server, then send the same prompt several times."""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
PORT = 8003
tok = AutoTokenizer.from_pretrained(MODEL)
recs = {}
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    recs[r["id"]] = r


def run(pid, steps=6):
    payload = {"model": MODEL, "prompt": tok.decode(recs[pid]["prompt_token_ids"]),
               "max_tokens": steps, "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())["token_ids"]


print("warmup A-0010:", run("A-0010", 4), flush=True)
for i in range(4):
    print(f"run{i}:", run("A-0006"), flush=True)
print("swap  A-0002:", run("A-0002"), flush=True)
print("again A-0006:", run("A-0006"), flush=True)
print("### WARM_DONE", flush=True)
