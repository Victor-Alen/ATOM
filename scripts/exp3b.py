#!/usr/bin/env python3
"""Inspect HF's MellumExperts parameter layout (hooks that return None)."""
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


def hook_mlp(m, i, o):
    cap["h2"] = i[0].detach()
    cap["moe"] = o.detach()


def hook_gate(m, i, o):
    if isinstance(o, tuple) and len(o) == 3:
        cap["sel"] = o[2].detach()
        cap["rw"] = o[1].detach()


mlp0 = model.model.layers[0].mlp
mlp0.register_forward_hook(hook_mlp)
mlp0.gate.register_forward_hook(hook_gate)

ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)
print("captured:", sorted(cap), flush=True)
h2 = cap["h2"].reshape(-1, cfg.hidden_size)
print("h2", tuple(h2.shape), "moe", tuple(cap["moe"].shape), "sel", tuple(cap["sel"].shape), flush=True)
exp = mlp0.experts
print("params:", [(n, tuple(p.shape)) for n, p in exp.named_parameters()], flush=True)
print("buffers:", [(n, tuple(b.shape)) for n, b in exp.named_buffers()], flush=True)
torch.save({"h2": h2.to(torch.bfloat16).cpu(), "moe": cap["moe"].reshape(-1, cfg.hidden_size).float().cpu(),
            "sel": cap["sel"].cpu(), "rw": cap["rw"].float().cpu()},
           "/workspace/logs/hf_moe_l0.pt")
print("### EXP3_DONE", flush=True)
