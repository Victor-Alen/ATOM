#!/usr/bin/env python3
"""Does forcing the same GEMM backend make the engine agree with HF?
Regenerates HF greedy tokens for one prompt in-process, then asks the engine."""
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

pid = "A-0006"
rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == pid:
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
print(f"hf_now {pid}: {out}", flush=True)
json.dump(out, open("/tmp/hf_a0006.json", "w"))
print("### HF_ONE_DONE", flush=True)
