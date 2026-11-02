#!/usr/bin/env python3
"""Fast C aggregate: 8 greedy tokens per prompt, single client, N prompts."""
import json
import sys
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 20

recs = []
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    recs.append(json.loads(line))
    if len(recs) >= LIMIT:
        break

tot = hit = exact = 0
for rec in recs:
    p = {"model": MODEL, "prompt": tok.decode(rec["prompt_token_ids"]), "max_tokens": STEPS,
         "temperature": 0.0, "ignore_eos": True, "stream": False}
    rq = urllib.request.Request("http://127.0.0.1:8003/v1/completions",
                                data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    try:
        got = json.loads(urllib.request.urlopen(rq, timeout=7200).read())["token_ids"]
    except Exception as exc:  # noqa: BLE001
        print(f"{rec['id']}: ERROR {exc!r}", flush=True)
        continue
    ref = rec["token_ids"][:STEPS]
    n = min(len(got), len(ref))
    m = sum(1 for a, b in zip(got, ref) if a == b)
    tot += n
    hit += m
    exact += int(n and m == n)
    print(f"{rec['id']} (len={len(rec['prompt_token_ids'])}): {m}/{n}", flush=True)
print(f"C aggregate: {hit}/{tot} = {hit/max(tot,1):.4f}  exact={exact}/{len(recs)}", flush=True)
print("### CFAST_DONE", flush=True)
