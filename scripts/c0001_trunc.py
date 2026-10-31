#!/usr/bin/env python3
"""Is C-0001's engine answer the HF answer for a truncated prefix?"""
import importlib.util, json, sys
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


ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()
print("HF ready", flush=True)

rec = [r for r in json.load(open("/tmp/three.json")) if r["id"] == "C-0001"][0]
full = rec["prompt_token_ids"]

for n in (8192, 12000, 12580):
    ids = torch.tensor([full[:n]], dtype=torch.long, device="cuda:0")
    out = []
    with torch.no_grad():
        for _ in range(8):
            logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
            nxt = int(logits.argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda:0")], dim=1)
    print(f"prefix={n}: {out}", flush=True)
print("engine     : [978, 1365, 1020, 1365, 2147, 669, 2677, 1277]", flush=True)
print("### TRUNC_DONE", flush=True)
