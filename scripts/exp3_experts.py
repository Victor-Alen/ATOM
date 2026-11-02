#!/usr/bin/env python3
"""Three-way expert GEMM comparison for layer 0 on A-0006:
(a) HF's own experts module, (b) grouped_mm, (c) per-expert plain linear."""
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
mlp0 = model.model.layers[0].mlp
mlp0.register_forward_hook(lambda m, i, o: (cap.setdefault("h2", i[0].detach()),
                                            cap.setdefault("moe", o.detach())))
gate_out = {}
mlp0.gate.register_forward_hook(
    lambda m, i, o: gate_out.setdefault("sel", o[2].detach()) if isinstance(o, tuple) and len(o) == 3 else None)

ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)

h2 = cap["h2"].reshape(-1, cfg.hidden_size)
moe_hf = cap["moe"].reshape(-1, cfg.hidden_size)
sel = gate_out["sel"]
print("h2", tuple(h2.shape), "moe", tuple(moe_hf.shape), "sel", tuple(sel.shape), flush=True)

exp = mlp0.experts
print("expert params:", [(n, tuple(p.shape)) for n, p in exp.named_parameters()], flush=True)
print("expert buffers:", [(n, tuple(b.shape)) for n, b in exp.named_buffers()], flush=True)
print("expert module type:", type(exp).__name__, flush=True)
print("### PROBE_DONE", flush=True)
