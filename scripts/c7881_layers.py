#!/usr/bin/env python3
"""Per-chunk, per-layer comparison at 7881 tokens; report the first bad chunk/layer."""
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
    if r["id"] == "C-0000":
        rec = r
        break
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
print("prompt tokens:", ids.shape[1], flush=True)

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

eng = torch.load("/workspace/logs/dumpc0/layers_0.pt")
for ci in (1, 2):
    first = None
    rows = []
    for li in range(28):
        k = f"c{ci}_blk_{li}"
        if k not in eng:
            continue
        d = (eng[k].float() - hs[li + 1]).abs().max().item()
        if d != 0.0 and first is None:
            first = li
        if li < 3 or d != 0.0:
            rows.append(f"L{li}={d:.6g}")
    print(f"chunk {ci}: first_bad_layer={first} " + " ".join(rows[:6]), flush=True)
print("hf top1:", int(out.logits[0, -1].argmax()), flush=True)
print("### C7881_LAYERS_DONE", flush=True)
