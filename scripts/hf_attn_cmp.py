#!/usr/bin/env python3
"""Capture HF's layer-0 attention residual output and compare with the engine dump."""
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

cap = {}
layers = model.model.layers


def pre_hook(name):
    def hook(module, args):
        cap[name] = args[0].detach()

    return hook


h1 = layers[0].post_attention_layernorm.register_forward_pre_hook(pre_hook("attn_res"))
h2 = layers[0].post_attention_layernorm.register_forward_hook(
    lambda m, i, o: cap.__setitem__("h2", o.detach())
)

rec = json.load(open("/tmp/a0012_run.json"))
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)
h1.remove()
h2.remove()

eng_attn = torch.load("/workspace/logs/dumpl/attn_L0.pt").float()
hf_attn = cap["attn_res"][0, -1].float().cpu()
print(f"L0 attention residual max|delta|={(eng_attn - hf_attn).abs().max().item():.6f}", flush=True)
print(f"  engine absmax={eng_attn.abs().max().item():.4f} hf absmax={hf_attn.abs().max().item():.4f}", flush=True)
torch.save(cap["attn_res"][0, -1].float().cpu(), "/workspace/logs/hf_attn_L0.pt")
print("### ATTN_CMP_DONE", flush=True)
