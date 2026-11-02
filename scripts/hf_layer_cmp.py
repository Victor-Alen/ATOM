#!/usr/bin/env python3
"""Compare HF's per-layer hidden (last prefill row) with the engine dumps."""
import importlib.util
import glob
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

rec = json.load(open("/tmp/a0012_run.json"))
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    out = model(input_ids=ids, logits_to_keep=1, output_hidden_states=True)
hs = out.hidden_states  # tuple: embeddings + 28 layer outputs
print("hf hidden layers:", len(hs), flush=True)

torch.save([h[0, -1].float().cpu() for h in hs], "/workspace/logs/hf_hs.pt")
print("saved hf_hs.pt", flush=True)
for li in range(len(hs) - 1):
    path = f"/workspace/logs/dumpl/blk_L{li}.pt"
    if not Path(path).exists():
        continue
    eng = torch.load(path).float()
    hf = hs[li + 1][0, -1].float().cpu()
    d = (eng - hf).abs().max().item()
    flag = "OK " if d == 0.0 else "DIFF"
    print(f"{flag} L{li:02d} max|delta|={d:.6f}", flush=True)
    if d != 0.0:
        break
print("### LAYER_CMP_DONE", flush=True)
