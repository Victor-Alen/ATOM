#!/usr/bin/env python3
"""Compare layer-0 attention internals: q post-norm, q/k post-rope, and the per-head output."""
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


def h_qn(m, i, o):
    cap.setdefault("q_norm", o.detach())


def h_o(m, i, o):
    cap.setdefault("o_in", i[0].detach())


a0 = model.model.layers[0].self_attn
a0.q_norm.register_forward_hook(h_qn)
a0.o_proj.register_forward_hook(h_o)
ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)

eng = torch.load("/workspace/logs/dumpl8/layers_0.pt")
qn = cap["q_norm"]
o_in = cap["o_in"]
print("HF q_norm shape", tuple(qn.shape), "o_in", tuple(o_in.shape), flush=True)

for name, ref_t in (("q0_norm", qn), ("o0_pre_o", o_in)):
    e = eng[name].float().cpu()
    f = ref_t.reshape(e.shape).to(torch.bfloat16).float().cpu() if ref_t.numel() == e.numel() else None
    if f is None:
        print(f"{name}: element count mismatch {e.numel()} vs {ref_t.numel()}", flush=True)
        continue
    d = (e - f).abs()
    per = d.reshape(d.shape[0], -1).amax(dim=-1)
    bad = (per > 0).nonzero().flatten()
    print(f"{name}: max={d.max().item():.6g} tokens={len(bad)}/{d.shape[0]} tokens={bad[:8].tolist()}", flush=True)

print("### INNER_CMP_DONE", flush=True)
