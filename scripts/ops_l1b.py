#!/usr/bin/env python3
"""Layer-1 sub-op check, minimal and robust: projections and q/k norm only.
If all match, the remaining suspect is rope."""
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")
EPS = 1e-6


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


eng = torch.load("/workspace/logs/dumpl5/layers_0.pt")
h1 = eng["h1_full"].to("cuda:0")
print("h1_full", tuple(h1.shape), flush=True)

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()
print("HF ready", flush=True)

cap = {}
attn1 = model.model.layers[1].self_attn
handles = [
    attn1.q_proj.register_forward_hook(lambda m, i, o: cap.setdefault("q_lin", o.detach())),
    attn1.k_proj.register_forward_hook(lambda m, i, o: cap.setdefault("k_lin", o.detach())),
    attn1.v_proj.register_forward_hook(lambda m, i, o: cap.setdefault("v_lin", o.detach())),
    attn1.q_norm.register_forward_hook(lambda m, i, o: cap.setdefault("q_norm", o.detach())),
    attn1.k_norm.register_forward_hook(lambda m, i, o: cap.setdefault("k_norm", o.detach())),
]
ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)
for h in handles:
    h.remove()

idx = json.load(open(Path(MODEL) / "model.safetensors.index.json"))
wm = idx["weight_map"]
base = "model.layers.1.self_attn."
w = {}
for nm, key in (("q_proj.weight", "wq"), ("k_proj.weight", "wk"), ("v_proj.weight", "wv"),
                ("q_norm.weight", "qn"), ("k_norm.weight", "kn")):
    with safe_open(Path(MODEL) / wm[base + nm], framework="pt") as fh:
        w[key] = fh.get_tensor(base + nm).to("cuda:0")


def rms(x, weight):
    f = x.float()
    v = f.pow(2).mean(-1, keepdim=True)
    return weight * (f * torch.rsqrt(v + EPS)).to(x.dtype)


nH, nKV, D = 32, 4, 128
for name, wkey, heads in (("q_lin", "wq", nH), ("k_lin", "wk", nKV), ("v_lin", "wv", nKV)):
    mine = F.linear(h1, w[wkey])
    hf = cap[name][0]
    mine = mine.reshape(hf.shape)
    print(f"{name}: manual vs HF max|delta| = {(mine.float() - hf.float()).abs().max().item():.6g}", flush=True)

qm = F.linear(h1, w["wq"]).view(-1, nH, D)
qm = rms(qm, w["qn"])
hfq = cap["q_norm"][0]
hfq = hfq.reshape(qm.shape) if hfq.numel() == qm.numel() else hfq
print(f"q_norm: manual vs HF max|delta| = {(qm.float() - hfq.float()).abs().max().item():.6g}", flush=True)
print("### OPS2_DONE", flush=True)
