#!/usr/bin/env python3
"""Full-matrix comparison of layer 1's input (post-LN) and layer 0's output:
which positions differ, not just the last row."""
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


eng = torch.load("/workspace/logs/dumpl5/layers_0.pt")
h1_eng = eng["h1_full"].float().cpu()
print("engine h1_full", tuple(h1_eng.shape), flush=True)

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

cap = {}
model.model.layers[1].input_layernorm.register_forward_hook(
    lambda m, i, o: cap.setdefault("h1_hf", o.detach()))
model.model.layers[0].register_forward_hook(
    lambda m, i, o: cap.setdefault("blk0_hf", (o[0] if isinstance(o, tuple) else o).detach()))
ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)

hf = cap["h1_hf"][0].float().cpu()
d = (h1_eng - hf).abs()
per_pos = d.amax(dim=-1)
bad = (per_pos > 0).nonzero().flatten()
print(f"L1 input: positions differing = {len(bad)} / {per_pos.numel()}", flush=True)
print(f"max|delta| overall = {d.max().item():.6g}", flush=True)
if len(bad):
    print("first differing positions:", bad[:8].tolist(), flush=True)
    print("last differing positions:", bad[-4:].tolist(), flush=True)
    print("last-row delta =", d[-1].max().item(), flush=True)
print("### FULL_CMP_DONE", flush=True)
