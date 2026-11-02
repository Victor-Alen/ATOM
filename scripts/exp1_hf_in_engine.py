#!/usr/bin/env python3
"""Experiment 1: run HF's own model INSIDE the ATOM engine environment (after importing
atom/aiter) and see whether it reproduces the standalone-HF token stream."""
import importlib.util
import json
import sys
from pathlib import Path

import atom  # noqa: F401  # replicate the engine's import side effects
import torch

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


print("atom imported; blas backend:", torch.backends.cuda.preferred_blas_library(), flush=True)
ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()
print("HF-in-engine ready", flush=True)

rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0006":
        rec = r
        break
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
out = []
with torch.no_grad():
    for _ in range(8):
        logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        nxt = int(logits.argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda:0")], dim=1)
print("hf_in_engine:", out, flush=True)
standalone = json.load(open("/workspace/correctness/hf_scenario_a_20_fresh.json"))["A-0006"][:8]
print("hf_standalone:", standalone, flush=True)
print("engine_ref   : [233, 233, 37, 26828, 23, 233, 2116, 895]", flush=True)
print("same as standalone:", out == standalone, flush=True)
print("### EXP1_DONE", flush=True)
