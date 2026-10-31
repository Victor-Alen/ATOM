#!/usr/bin/env python3
"""Run the reference path over scenario prompts and compare with HF token ids."""
import json
import sys
import time
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
BASE = "/workspace/correctness"

CASES = [
    ("A", "hf_scenario_a_20_eager.jsonl", 5),
    ("B", "hf_scenario_b_20_eager.jsonl", 5),
    ("C", "hf_scenario_c_200_eager.jsonl", 5),
]
STEPS = 64

tok = AutoTokenizer.from_pretrained(MODEL)


def gen(text, steps):
    payload = {
        "model": MODEL,
        "prompt": text,
        "max_tokens": steps,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8001/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())["token_ids"]


for label, fname, limit in CASES:
    recs = []
    with open(f"{BASE}/{fname}", encoding="utf-8") as fh:
        for line in fh:
            recs.append(json.loads(line))
            if len(recs) >= limit:
                break
    matched = total = 0
    exact = 0
    t_case = time.time()
    for rec in recs:
        ids = rec["prompt_token_ids"]
        text = tok.decode(ids)
        if tok.encode(text, add_special_tokens=False) != ids:
            print(f"  {rec['id']}: tokenizer round-trip failed, skipped", flush=True)
            continue
        got = gen(text, STEPS)
        ref = rec["token_ids"][:STEPS]
        n = min(len(got), len(ref))
        same = sum(1 for i in range(n) if got[i] == ref[i])
        matched += same
        total += n
        if same == n:
            exact += 1
        print(f"  {rec['id']}: {same}/{n} (len {len(ids)})", flush=True)
    rate = matched / total if total else 0.0
    print(
        f"{label}: prompts={len(recs)} agree={rate:.4f} exact={exact}/{len(recs)} "
        f"elapsed={time.time() - t_case:.0f}s",
        flush=True,
    )
print("### SCEN_DONE", flush=True)
