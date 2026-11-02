#!/usr/bin/env python3
"""Position-wise comparison of layer-0's attention output: engine vs HF."""
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
    cap["attn0"] = (o[0] if isinstance(o, tuple) else o).detach()


model.model.layers[0].self_attn.register_forward_hook(hook)
ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)

eng = torch.load("/workspace/logs/dumpl6/layers_0.pt")["attn0_full"].float().cpu()
hf = cap["attn0"][0].float().cpu()
d = (eng - hf).abs()
per = d.amax(dim=-1)
bad = (per > 0).nonzero().flatten()
print(f"positions differing = {len(bad)} / {per.numel()}", flush=True)
print(f"max|delta| = {d.max().item():.6g}", flush=True)
print("first:", bad[:10].tolist(), flush=True)
print("last :", bad[-5:].tolist(), flush=True)
print("last-row delta =", d[-1].max().item(), flush=True)
torch.save(bad.cpu(), "/workspace/logs/bad_positions.pt")
print("### ATTN0_CMP_DONE", flush=True)
