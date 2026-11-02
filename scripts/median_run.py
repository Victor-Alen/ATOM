#!/usr/bin/env python3
"""A/B/C greedy agreement, judge protocol: 3 runs per prompt, median taken."""
import json
import statistics
import time
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
PORT = 8003
BASE = "/workspace/correctness"
STEPS = 64
RUNS = 3
MAXLEN = 12000

tok = AutoTokenizer.from_pretrained(MODEL)


def gen(text, steps=STEPS):
    payload = {"model": MODEL, "prompt": text, "max_tokens": steps,
               "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())["token_ids"]


def score(label, recs, ref_of):
    per_prompt = []
    t0 = time.time()
    for rec in recs:
        ref = ref_of(rec)[:STEPS]
        text = tok.decode(rec["prompt_token_ids"])
        ags = []
        for _ in range(RUNS):
            try:
                got = gen(text)
            except Exception as exc:  # noqa: BLE001
                print(f"  {rec['id']}: ERR {type(exc).__name__}", flush=True)
                continue
            n = min(len(got), len(ref))
            ags.append(sum(1 for i in range(n) if got[i] == ref[i]) / n if n else 0.0)
        if not ags:
            continue
        med = statistics.median(ags)
        per_prompt.append(med)
        print(f"  {rec['id']}: runs={[round(a,3) for a in ags]} median={med:.3f}", flush=True)
    overall = sum(per_prompt) / len(per_prompt) if per_prompt else 0.0
    print(f"{label}: prompts={len(per_prompt)} median-agreement={overall:.4f} "
          f"prompts>=0.9: {sum(1 for m in per_prompt if m >= 0.9)}/{len(per_prompt)} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)


fresh_a = json.load(open(f"{BASE}/hf_scenario_a_20_fresh.json"))
a_recs = [json.loads(l) for l in open(f"{BASE}/hf_scenario_a_20_eager.jsonl", encoding="utf-8")]
score("A", a_recs, lambda r: fresh_a[r["id"]])

b_recs = [json.loads(l) for l in open(f"{BASE}/hf_scenario_b_20_eager.jsonl", encoding="utf-8")]
score("B", b_recs, lambda r: r["token_ids"])

c_recs = []
for line in open(f"{BASE}/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if len(r["prompt_token_ids"]) <= MAXLEN:
        c_recs.append(r)
    if len(c_recs) >= 20:
        break
score("C", c_recs, lambda r: r["token_ids"])
print("### MEDIAN_DONE", flush=True)
