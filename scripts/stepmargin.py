#!/usr/bin/env python3
"""Margin at every generated step for one A prompt: how many steps sit inside the
kernel error band (0.03-0.14) and therefore flip no matter what."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")
STEPS = 10


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0006":
        rec = r
        break

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

t = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
margins = []
with torch.no_grad():
    for step in range(STEPS):
        lg = model(input_ids=t, logits_to_keep=1).logits[0, -1].float()
        top2 = torch.topk(lg, 2)
        m = float(top2.values[0] - top2.values[1])
        margins.append(m)
        print(f"step {step}: top1={int(top2.indices[0])} margin={m:.5f} "
              f"{'FLIPPABLE' if m < 0.14 else ''}", flush=True)
        t = torch.cat([t, torch.tensor([[int(top2.indices[0])]], device="cuda:0")], dim=1)

inside = sum(1 for m in margins if m < 0.14)
print(f"margins: {[round(m,3) for m in margins]}", flush=True)
print(f"steps inside the 0.14 error band: {inside}/{STEPS}", flush=True)
print("### STEPMARGIN_DONE", flush=True)
