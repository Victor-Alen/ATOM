#!/usr/bin/env python3
"""Experiment 2: HF per-layer hidden states for A-0006, computed inside the engine env."""
import importlib.util
import json
import sys
from pathlib import Path

import atom  # noqa: F401
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

rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0006":
        rec = r
        break
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    out = model(input_ids=ids, logits_to_keep=1, output_hidden_states=True)
hs = out.hidden_states
rows = [h[0, -1].float().cpu() for h in hs]
torch.save({"layers": rows, "logits_top1": int(out.logits[0, -1].argmax())},
           "/workspace/logs/hf_a0006_layers.pt")
print("saved", len(rows), "layer rows; top1", int(out.logits[0, -1].argmax()), flush=True)
print("### EXP2_DONE", flush=True)
