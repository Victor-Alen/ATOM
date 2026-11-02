#!/usr/bin/env python3
"""C-0011: engine logits for step 0 and 1 vs HF's, same positions."""
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


rec = None
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "C-0011":
        rec = r
        break
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

with torch.no_grad():
    lg0 = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float().cpu()
    t0 = int(lg0.argmax())
    ids1 = torch.cat([ids, torch.tensor([[t0]], device="cuda:0")], dim=1)
    lg1 = model(input_ids=ids1, logits_to_keep=1).logits[0, -1].float().cpu()

d = Path("/workspace/logs/logitd")
eng = sorted(d.glob("logits_*.pt")) if d.exists() else []
print("engine dumps:", [p.name for p in eng], flush=True)
if eng:
    e0 = torch.load(eng[0]).float()
    d0 = (e0 - lg0).abs().max().item()
    print(f"step0: hf_argmax={t0} eng_argmax={int(e0.argmax())} max|delta|={d0:.6g}", flush=True)
if len(eng) > 1:
    e1 = torch.load(eng[1]).float()
    d1 = (e1 - lg1).abs().max().item()
    print(f"step1: hf_argmax={int(lg1.argmax())} eng_argmax={int(e1.argmax())} max|delta|={d1:.6g}", flush=True)
print("### LOGITS_DONE", flush=True)
