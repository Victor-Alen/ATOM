#!/usr/bin/env python3
"""Sanity check for scenario C: is the recorded reference itself reproducible by HF?"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "C-0002"


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


rec = None
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == TARGET:
        rec = r
        break
print("prompt tokens:", len(rec["prompt_token_ids"]), "ref tokens:", rec["token_ids"][:8], flush=True)

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
out = []
with torch.no_grad():
    for _ in range(8):
        lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        nxt = int(lg.argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda:0")], dim=1)
print("hf_fresh   :", out, flush=True)
print("recorded   :", rec["token_ids"][:8], flush=True)
print("match:", out == rec["token_ids"][:8], flush=True)
print("### CSCAN_DONE", flush=True)
