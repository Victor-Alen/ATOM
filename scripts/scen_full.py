#!/usr/bin/env python3
"""Run scenarios A/B/C through the exact reference path (64 greedy steps)."""
import json
import sys
import time
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
BASE = "/workspace/correctness"
PORT = 8001
STEPS = 64
MAXLEN = 12000

tok = AutoTokenizer.from_pretrained(MODEL)


def gen(text, steps=STEPS):
    payload = {"model": MODEL, "prompt": text, "max_tokens": steps,
               "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        return json.loads(resp.read())["token_ids"]


CASES = [("A", "hf_scenario_a_20_eager.jsonl", 20),
         ("B", "hf_scenario_b_20_eager.jsonl", 20),
         ("C", "hf_scenario_c_200_eager.jsonl", 40)]

for label, fname, limit in CASES:
    recs = []
    with open(f"{BASE}/{fname}", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if len(r["prompt_token_ids"]) <= MAXLEN:
                recs.append(r)
            if len(recs) >= limit:
                break
    matched = total = 0
    exact = 0
    skipped = 0
    t0 = time.time()
    for rec in recs:
        ids = rec["prompt_token_ids"]
        try:
            got = gen(tok.decode(ids))
        except Exception as exc:  # noqa: BLE001
            print(f"  {rec['id']}: ERROR {type(exc).__name__}", flush=True)
            skipped += 1
            continue
        ref = rec["token_ids"][:STEPS]
        n = min(len(got), len(ref))
        same = sum(1 for i in range(n) if got[i] == ref[i])
        matched += same
        total += n
        if same == n:
            exact += 1
        print(f"  {rec['id']}: {same}/{n} len={len(ids)}", flush=True)
    rate = matched / total if total else 0.0
    print(f"{label}: prompts={len(recs)} agree={rate:.4f} exact={exact}/{len(recs)} "
          f"errors={skipped} elapsed={time.time()-t0:.0f}s", flush=True)
print("### SCEN_REF_DONE", flush=True)
