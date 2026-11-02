#!/usr/bin/env python3
"""HumanEval pass@1 against the running ATOM server, proper test harness."""
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8003
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 164
OUT = sys.argv[3] if len(sys.argv) > 3 else "/workspace/logs/humaneval_atom.json"

tasks = [json.loads(l) for l in open("/data/datasets/HumanEval.jsonl", encoding="utf-8")]


def generate(prompt, max_tokens=256):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())["choices"][0]["text"]


def truncate(completion):
    for marker in ("\nclass ", "\ndef ", "\nif __name__", "\nprint(", "\n# Test"):
        idx = completion.find(marker)
        if idx != -1:
            completion = completion[:idx]
    return completion


def check(task, completion):
    program = task["prompt"] + completion + "\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "program.py"
        p.write_text(program, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(p)], capture_output=True, timeout=20, cwd=d)
            return r.returncode == 0, r.stderr.decode(errors="replace")[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


results = []
passed = 0
for task in tasks[:LIMIT]:
    try:
        comp = truncate(generate(task["prompt"]))
        ok, err = check(task, comp)
    except Exception as exc:  # noqa: BLE001
        ok, err = False, f"{type(exc).__name__}: {exc}"
    results.append({"task_id": task["task_id"], "passed": ok, "error": err})
    passed += ok
    print(f"{task['task_id']}: {'PASS' if ok else 'fail'} ({passed}/{len(results)})", flush=True)

rate = passed / len(results) if results else 0.0
print(f"pass@1 = {passed}/{len(results)} = {rate:.4f}", flush=True)
json.dump({"count": len(results), "passed": passed, "pass_at_1": rate, "results": results},
          open(OUT, "w"))
print("### HE_DONE", flush=True)
