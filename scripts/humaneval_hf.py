#!/usr/bin/env python3
"""HumanEval pass@1 for the HF reference, same prompts and harness as the ATOM run."""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")
OUT = "/workspace/logs/he_hf_164.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 164


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()
tok = __import__("transformers").AutoTokenizer.from_pretrained(MODEL)
print("HF ready", flush=True)

tasks = [json.loads(l) for l in open("/data/datasets/HumanEval.jsonl", encoding="utf-8")]


def generate(prompt, max_new=256):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


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
print(f"HF pass@1 = {passed}/{len(results)} = {rate:.4f}", flush=True)
json.dump({"count": len(results), "passed": passed, "pass_at_1": rate, "results": results},
          open(OUT, "w"))
print("### HE_HF_DONE", flush=True)
