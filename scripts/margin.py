#!/usr/bin/env python3
"""HF top1-top2 margin at the first generated token for A vs B prompts.
Small margins mean a 1-ulp kernel difference necessarily flips the token."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


def recs(path, n):
    out = []
    for line in open(path, encoding="utf-8"):
        out.append(json.loads(line))
        if len(out) >= n:
            break
    return out


A = recs("/workspace/correctness/hf_scenario_a_20_eager.jsonl", 6)
B = recs("/workspace/correctness/hf_scenario_b_20_eager.jsonl", 6)

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

def margin(rec):
    ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
    top2 = torch.topk(lg, 2)
    return float(top2.values[0] - top2.values[1]), int(top2.indices[0])

for label, group in (("A", A), ("B", B)):
    vals = []
    for rec in group:
        m, t = margin(rec)
        vals.append(m)
        print(f"{label} {rec['id']} len={len(rec['prompt_token_ids'])} top1={t} margin={m:.5f}", flush=True)
    vals.sort()
    print(f"{label}: margins sorted = {[round(v,4) for v in vals]}", flush=True)
    print(f"{label}: median={vals[len(vals)//2]:.5f} min={vals[0]:.5f}", flush=True)
print("### MARGIN_DONE", flush=True)
