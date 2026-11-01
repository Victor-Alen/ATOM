#!/usr/bin/env python3
"""Is the prompt the engine sees identical to the reference prompt?
Also test whether the server accepts raw token ids."""
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
tok = AutoTokenizer.from_pretrained(MODEL)

scores = {
    "A-0009": 64, "A-0010": 64, "A-0017": 64,
    "A-0012": 2, "A-0006": 5, "A-0016": 2, "A-0004": 7,
}
recs = {}
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] in scores:
        recs[r["id"]] = r

for pid, rec in recs.items():
    ids = rec["prompt_token_ids"]
    text = tok.decode(ids)
    back = tok.encode(text, add_special_tokens=False)
    same = back == ids
    n = min(len(ids), len(back))
    diff = next((i for i in range(n) if ids[i] != back[i]), None)
    print(f"{pid}: agree_tokens={scores[pid]:2d} roundtrip={same} len={len(ids)}->{len(back)} first_diff={diff}", flush=True)

# does the API accept token ids directly?
rec = recs["A-0012"]
payload = {
    "model": MODEL,
    "prompt": rec["prompt_token_ids"],
    "max_tokens": 2,
    "temperature": 0.0,
    "ignore_eos": True,
    "stream": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        print("ids prompt accepted:", json.loads(resp.read())["token_ids"], flush=True)
except Exception as exc:  # noqa: BLE001
    print("ids prompt failed:", type(exc).__name__, str(exc)[:120], flush=True)
print("### RT_DONE", flush=True)
