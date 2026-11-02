#!/usr/bin/env python3
"""Overnight harness: reference path (8001) and production baseline (8002) on A/B/C."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
OUT = Path("/workspace/logs/overnight")
OUT.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)

SETS = {
    "A": ("/workspace/correctness/hf_scenario_a_20_eager.jsonl", None),
    "B": ("/workspace/correctness/hf_scenario_b_20_eager.jsonl", None),
    "C": ("/workspace/correctness/hf_scenario_c_200_eager.jsonl", None),
}
FRESH_A = json.load(open("/workspace/correctness/hf_scenario_a_20_fresh.json"))


def wait_health(port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(5)
    return False


def gen(port, text, steps=64):
    p = {"model": MODEL, "prompt": text, "max_tokens": steps, "temperature": 0.0,
         "ignore_eos": True, "stream": False}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=7200) as r:
        return json.loads(r.read())["token_ids"]


def run(port, label, name, limit=None):
    path, _ = SETS[name]
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    if limit:
        recs = recs[:limit]
    out = OUT / f"{label}_{name}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    tot_tok = tot_hit = exact = 0
    for rec in recs:
        if rec["id"] in done:
            continue
        ref = FRESH_A[rec["id"]] if name == "A" else rec["token_ids"]
        ref = ref[:64]
        try:
            got = gen(port, tok.decode(rec["prompt_token_ids"]))
            same = sum(1 for a, b in zip(got, ref) if a == b)
            n = min(len(got), len(ref))
            row = {"id": rec["id"], "same": same, "n": n, "got": got, "ref": ref}
        except Exception as exc:  # noqa: BLE001
            row = {"id": rec["id"], "error": repr(exc)}
            n = 0
            same = 0
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        tot_tok += n
        tot_hit += same
        exact += int(n > 0 and same == n)
        print(f"{label} {name} {rec['id']}: {same}/{n}", flush=True)
    print(f"== {label} {name} SUMMARY rate={tot_hit/max(tot_tok,1):.4f} "
          f"exact={exact} tokens={tot_hit}/{tot_tok}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    port = int(sys.argv[2])
    if not wait_health(port):
        print("server not healthy", flush=True)
        sys.exit(1)
    print(f"healthy {port}", flush=True)
    for name in which.split(","):
        run(port, sys.argv[3], name, int(sys.argv[4]) if len(sys.argv) > 4 else None)
    print("### OVERNIGHT_DONE", flush=True)
