#!/usr/bin/env python3
"""HF logits for A-0012 at the same steps the engine logged."""
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
ids = rec["prompt_token_ids"]
feed = list(ids)
with torch.no_grad():
    for step in range(3):
        t = torch.tensor([feed], dtype=torch.long, device="cuda:0")
        logits = model(input_ids=t, logits_to_keep=1).logits[0, -1].float()
        v, i = torch.topk(logits, 5)
        print(
            f"hf   step{step}: argmax={int(i[0])} top5={[int(x) for x in i.tolist()]} "
            f"vals={[round(float(x), 4) for x in v.tolist()]} "
            f"margin={float(v[0]-v[1]):.4f}",
            flush=True,
        )
        feed.append(int(i[0]))
print("### HF_STEPS_DONE", flush=True)
