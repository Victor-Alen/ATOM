#!/usr/bin/env python3
"""Which layer first diverges at 1200 tokens (last row, engine vs HF)."""
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
with torch.no_grad():
    out = model(input_ids=ids, logits_to_keep=1, output_hidden_states=True)
hs = [h[0, -1].float().cpu() for h in out.hidden_states]
eng = torch.load("/workspace/logs/dumpc/layers_0.pt")
first = None
for li in range(28):
    k = f"blk_{li}"
    if k not in eng:
        continue
    d = (eng[k].float() - hs[li + 1]).abs().max().item()
    if d != 0.0 and first is None:
        first = li
    if li < 8 or d != 0.0:
        print(f"L{li:02d}: {d:.6g}", flush=True)
print("first differing layer:", first, flush=True)
print("layer_types:", list(model.config.layer_types)[:8], flush=True)
print("### C1200_LAYERS_DONE", flush=True)
