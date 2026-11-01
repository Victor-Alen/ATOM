#!/usr/bin/env python3
"""Localise L0: (a) rope formula/broadcast, (b) norm weight/eps."""
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
print("HF ready", flush=True)

modeling = sys.modules["transformers.models.mellum.modeling_mellum"]
orig = modeling.apply_rotary_pos_emb
cap = {"n": 0}


def wrapped(q, k, cos, sin, unsqueeze_dim=1, *a, **kw):
    if cap["n"] == 0:
        cap.update(q=q.detach(), k=k.detach(), cos=cos.detach(), sin=sin.detach())
    cap["n"] += 1
    return orig(q, k, cos, sin, unsqueeze_dim, *a, **kw)


modeling.apply_rotary_pos_emb = wrapped
hcap = {}
model.model.layers[0].input_layernorm.register_forward_hook(
    lambda m, i, o: hcap.setdefault("h", o.detach())
)

rec = json.load(open("/tmp/a0012_run.json"))
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)
modeling.apply_rotary_pos_emb = orig

# (a) rope: HF's own output vs my formula using the same cos/sin
q_in, k_in = cap["q"], cap["k"]
hf_q, hf_k = orig(q_in, k_in, cap["cos"], cap["sin"], 1)
cos1 = cap["cos"][0].unsqueeze(1)
sin1 = cap["sin"][0].unsqueeze(1)
q3 = q_in[0].transpose(0, 1)  # [S, heads, dim]
k3 = k_in[0].transpose(0, 1)
my_q = q3 * cos1 + modeling.rotate_half(q3) * sin1
my_k = k3 * cos1 + modeling.rotate_half(k3) * sin1
print(f"rope formula max|delta| q={float((my_q - hf_q[0].transpose(0,1)).abs().max()):.6f} "
      f"k={float((my_k - hf_k[0].transpose(0,1)).abs().max()):.6f}", flush=True)

# (b) norm: HF norm applied to the engine embedding row vs the engine's h_ln dump
emb = torch.load("/workspace/logs/dumpl/emb.pt").float().cuda()
w = model.model.layers[0].input_layernorm.weight
eps = model.model.layers[0].input_layernorm.variance_epsilon
with torch.no_grad():
    hf_norm = model.model.layers[0].input_layernorm(emb.to(torch.bfloat16))
eng_h = torch.load("/workspace/logs/dumpl/h_ln_L0.pt").float().cuda()
print(f"norm(engine emb) vs engine h_ln max|delta|={float((hf_norm.float()-eng_h).abs().max()):.6f}", flush=True)
print(f"eps used by HF = {eps}", flush=True)
print("### LOCALISE_DONE", flush=True)
