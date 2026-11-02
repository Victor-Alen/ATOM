#!/usr/bin/env python3
"""HF layer-0 gate top-8 for A-0006, to compare with the engine's selection."""
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
model.model.layers[0].mlp.gate.register_forward_hook(
    lambda m, i, o: cap.setdefault("gate", (o[0] if isinstance(o, tuple) else o).detach()))
ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)
g = cap["gate"][0].float()
probs = torch.softmax(g, dim=-1)
topv, topi = torch.topk(probs, 8, dim=-1)
torch.save({"gate_logits": g.cpu(), "topi": topi.cpu(), "topv": topv.cpu()},
           "/workspace/logs/gate_hf.pt")
print("hf gate saved", tuple(g.shape), flush=True)

eng = torch.load("/workspace/logs/gate_engine.pt")
e_topi = eng["topi"]
h_topi = topi.cpu()
same = (e_topi == h_topi).all(dim=-1)
print(f"positions with identical top8 = {int(same.sum())} / {same.numel()}", flush=True)
bad = (~same).nonzero().flatten().tolist()
print("positions differing:", bad[:20], flush=True)
dl = (eng["gate_logits"].float() - g.cpu()).abs().max().item()
print(f"gate logits max|delta| = {dl:.6g}", flush=True)
print("### GATE_CMP_DONE", flush=True)
