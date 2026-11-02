#!/usr/bin/env python3
"""Compare layer-0 tensors at 1200 tokens (sliding window engaged)."""
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
    if r["id"] == "C-0002":
        rec = r
        break
ids = torch.tensor([rec["prompt_token_ids"][:1200]], dtype=torch.long, device="cuda:0")

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


def h_mlp_in(m, i, o):
    cap.setdefault("h2", i[0].detach())


def h_o(m, i, o):
    cap.setdefault("o_in", i[0].detach())


def h_attn(m, i, o):
    cap.setdefault("attn", (o[0] if isinstance(o, tuple) else o).detach())


a0 = model.model.layers[0].self_attn
a0.q_norm.register_forward_hook(h_qn)
a0.o_proj.register_forward_hook(h_o)
a0.register_forward_hook(h_attn)
model.model.layers[0].mlp.register_forward_hook(h_mlp_in)

with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)

eng = torch.load("/workspace/logs/dumpc/layers_0.pt")
for name, cap_key in (("q0_norm", "q_norm"), ("attn0_full", "attn"), ("o0_pre_o", "o_in")):
    e = eng[name].float().cpu()
    t = cap[cap_key].reshape(e.shape).float().cpu()
    d = (e - t).abs()
    per = d.reshape(d.shape[0], -1).amax(dim=-1)
    bad = (per > 0).nonzero().flatten()
    print(f"{name}: max={d.max().item():.6g} tokens={len(bad)}/{d.shape[0]} first={bad[:6].tolist()}", flush=True)
print("### C1200_CMP_DONE", flush=True)
