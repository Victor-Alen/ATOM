#!/usr/bin/env python3
"""Gate run with the current code: A against fresh HF refs, B against recorded refs,
C (<=12k) against recorded refs."""
import json
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


def score(label, recs, ref_of):
    matched = total = exact = 0
    t0 = time.time()
    for rec in recs:
        ref = ref_of(rec)[:STEPS]
        try:
            got = gen(tok.decode(rec["prompt_token_ids"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  {rec['id']}: ERROR {type(exc).__name__}", flush=True)
            continue
        n = min(len(got), len(ref))
        same = sum(1 for i in range(n) if got[i] == ref[i])
        matched += same
        total += n
        exact += same == n
        print(f"  {rec['id']}: {same}/{n} len={len(rec['prompt_token_ids'])}", flush=True)
    rate = matched / total if total else 0.0
    print(f"{label}: prompts={len(recs)} agree={rate:.4f} exact={exact}/{len(recs)} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)


fresh_a = json.load(open(f"{BASE}/hf_scenario_a_20_fresh.json"))
a_recs = [json.loads(l) for l in open(f"{BASE}/hf_scenario_a_20_eager.jsonl", encoding="utf-8")]
score("A(fresh)", a_recs, lambda r: fresh_a[r["id"]])

b_recs = [json.loads(l) for l in open(f"{BASE}/hf_scenario_b_20_eager.jsonl", encoding="utf-8")]
score("B", b_recs, lambda r: r["token_ids"])

c_recs = []
for line in open(f"{BASE}/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if len(r["prompt_token_ids"]) <= MAXLEN:
        c_recs.append(r)
    if len(c_recs) >= 40:
        break
score("C", c_recs, lambda r: r["token_ids"])
print("### GATE_DONE", flush=True)
