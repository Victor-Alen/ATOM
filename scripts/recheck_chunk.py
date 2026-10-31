#!/usr/bin/env python3
"""Re-check the prompts that failed before the chunk-safe history fix."""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
BASE = "/workspace/correctness"
STEPS = 8

tok = AutoTokenizer.from_pretrained(MODEL)


def gen(text, steps=STEPS):
    payload = {"model": MODEL, "prompt": text, "max_tokens": steps,
               "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8001/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())["token_ids"]


CASES = [("hf_scenario_a_20_eager.jsonl", ["A-0000", "A-0001", "A-0002", "A-0003", "A-0004"]),
         ("hf_scenario_c_200_eager.jsonl", ["C-0001", "C-0003"])]

for fname, ids_wanted in CASES:
    recs = {}
    with open(f"{BASE}/{fname}", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] in ids_wanted:
                recs[r["id"]] = r
    for pid in ids_wanted:
        rec = recs.get(pid)
        if not rec:
            continue
        ids = rec["prompt_token_ids"]
        text = tok.decode(ids)
        got = gen(text)
        ref = rec["token_ids"][:STEPS]
        same = sum(1 for i in range(min(len(got), len(ref))) if got[i] == ref[i])
        print(f"{pid}: {same}/{STEPS} len={len(ids)}", flush=True)
        if same < STEPS:
            print(f"   atom={got[:8]}", flush=True)
            print(f"   hf  ={ref[:8]}", flush=True)
print("### CHECK_DONE", flush=True)
