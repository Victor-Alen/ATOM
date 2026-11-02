#!/usr/bin/env python3
"""Compare layer-0 expert selection: engine vs HF (HF gate returns 3 values)."""
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


ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

cap = {}


def hook(m, i, o):
    if isinstance(o, tuple) and len(o) == 3:
        cap.setdefault("weights", o[1].detach().cpu())
        cap.setdefault("experts", o[2].detach().cpu())


model.model.layers[0].mlp.gate.register_forward_hook(hook)
ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)
print("captured:", sorted(cap), flush=True)

h_exp = cap["experts"]
print("hf experts shape", tuple(h_exp.shape), flush=True)
eng = torch.load("/workspace/logs/gate_engine.pt")
e_exp = eng["topi"]
print("engine topi shape", tuple(e_exp.shape), flush=True)
same = (e_exp == h_exp).all(dim=-1) if e_exp.shape == h_exp.shape else None
if same is not None:
    print(f"positions with identical expert set = {int(same.sum())} / {same.numel()}", flush=True)
    bad = (~same).nonzero().flatten().tolist()
    print("positions differing:", bad[:20], flush=True)
w = cap["weights"].float()
ew = eng["topv"].float()
if ew.shape == w.shape:
    print(f"routing weights max|delta| = {(ew - w).abs().max().item():.6g}", flush=True)
print("### GATE_CMP2_DONE", flush=True)
