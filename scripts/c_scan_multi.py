#!/usr/bin/env python3
"""How many scenario-C reference entries does HF itself reproduce?"""
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


recs = [json.loads(l) for l in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8")]
ids = sys.argv[1:] or ["C-0000", "C-0001", "C-0002", "C-0003", "C-0004"]
want = [r for r in recs if r["id"] in ids]

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

for rec in want:
    ids_t = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
    out = []
    with torch.no_grad():
        for _ in range(8):
            lg = model(input_ids=ids_t, logits_to_keep=1).logits[0, -1]
            nxt = int(lg.argmax())
            out.append(nxt)
            ids_t = torch.cat([ids_t, torch.tensor([[nxt]], device="cuda:0")], dim=1)
    rec_ref = rec["token_ids"][:8]
    n = 0
    for a, b in zip(out, rec_ref):
        if a != b:
            break
        n += 1
    print(f"{rec['id']}: len={len(rec['prompt_token_ids'])} prefix_match={n}/8 "
          f"{'OK' if n == 8 else 'MISMATCH'} fresh={out[:5]} rec={rec_ref[:5]}", flush=True)
print("### SCAN_DONE", flush=True)
