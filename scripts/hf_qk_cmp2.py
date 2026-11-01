#!/usr/bin/env python3
"""Compare engine L0 q/k (post rope, prefill row) with HF's layer-0 values only."""
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
        cap["q"] = q.detach()
        cap["k"] = k.detach()
        cap["cos"] = cos.detach()
        cap["sin"] = sin.detach()
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
print(f"apply_rotary calls seen: {cap['n']}", flush=True)

eng_h = torch.load("/workspace/logs/dumpl/h_ln_L0.pt").float()
hf_h = hcap["h"][0, -1].float().cpu()
print(f"post-LN hidden max|delta|={(eng_h - hf_h).abs().max().item():.6f}", flush=True)

q_rope, k_rope = orig(cap["q"], cap["k"], cap["cos"], cap["sin"], 1)
hq = q_rope[0, :, -1, :].float().cpu()
hk = k_rope[0, :, -1, :].float().cpu()
eq = torch.load("/workspace/logs/dumpl/q_L0.pt").float()
ek = torch.load("/workspace/logs/dumpl/k_L0.pt").float()
print(f"q post-rope max|delta|={(eq - hq).abs().max().item():.6f}", flush=True)
print(f"k post-rope max|delta|={(ek - hk).abs().max().item():.6f}", flush=True)
print("### QK_CMP2_DONE", flush=True)
