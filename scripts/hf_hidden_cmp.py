#!/usr/bin/env python3
"""HF hidden state at the last prefill position for A-0012, for a direct diff."""
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

rec = json.load(open("/tmp/a0012_run.json"))
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    out = model(input_ids=ids, logits_to_keep=1, output_hidden_states=True)
    hs = out.hidden_states[-1][0, -1].float().cpu()
    torch.save(hs, "/workspace/logs/hf_hidden.pt")
    print("hf hidden saved", hs.shape, float(hs.abs().max()), flush=True)

eng = torch.load("/workspace/logs/engine_hidden.pt").float()
print("engine len", eng.numel(), flush=True)
n = min(eng.numel(), hs.numel())
d = (eng[:n] - hs[:n]).abs()
print(f"max|delta|={float(d.max()):.6f} equal={bool(torch.equal(eng[:n], hs[:n]))}", flush=True)
print("### HIDDEN_CMP_DONE", flush=True)
