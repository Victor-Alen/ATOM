#!/usr/bin/env python3
"""HF sub-operation values for layers 0-3 (last row of A-0006 prefill)."""
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
layers = model.model.layers
handles = []
for li in range(4):
    lay = layers[li]
    handles.append(lay.input_layernorm.register_forward_hook(
        lambda m, i, o, li=li: cap.__setitem__(f"h_{li}", o.detach()[0, -1].float().cpu())))
    handles.append(lay.self_attn.register_forward_hook(
        lambda m, i, o, li=li: cap.__setitem__(
            f"attn_out_{li}", (o[0] if isinstance(o, tuple) else o).detach()[0, -1].float().cpu())))
    handles.append(lay.post_attention_layernorm.register_forward_hook(
        lambda m, i, o, li=li: cap.__setitem__(f"h2_{li}", o.detach()[0, -1].float().cpu())))
    handles.append(lay.mlp.register_forward_hook(
        lambda m, i, o, li=li: cap.__setitem__(
            f"moe_{li}", (o[0] if isinstance(o, tuple) else o).detach()[0, -1].float().cpu())))

rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0006":
        rec = r
        break
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)
for h in handles:
    h.remove()
torch.save(cap, "/workspace/logs/hf_subops.pt")
print("saved keys:", sorted(cap)[:8], flush=True)
print("### SUBOPS_DONE", flush=True)
