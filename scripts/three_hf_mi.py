#!/usr/bin/env python3
"""Recompute HF greedy tokens for the three mismatching prompts (on the MI308X node)."""
import importlib.util, json, sys
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

for rec in json.load(open("/tmp/three.json")):
    ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
    out = []
    with torch.no_grad():
        for _ in range(8):
            logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
            nxt = int(logits.argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda:0")], dim=1)
    rec_ref = rec["token_ids"][:8]
    print(f"{rec['id']}: len={len(rec['prompt_token_ids'])}", flush=True)
    print(f"   hf_now   ={out}", flush=True)
    print(f"   recorded ={rec_ref}  same={out == rec_ref}", flush=True)
print("### THREE_DONE", flush=True)
