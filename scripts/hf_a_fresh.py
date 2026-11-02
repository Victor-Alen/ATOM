#!/usr/bin/env python3
"""Regenerate HF greedy tokens (64) for the 20 scenario-A prompts."""
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

out = {}
recs = [json.loads(l) for l in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8")]
for rec in recs[:20]:
    ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")
    toks = []
    with torch.no_grad():
        for _ in range(64):
            logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
            nxt = int(logits.argmax())
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda:0")], dim=1)
    out[rec["id"]] = toks
    print(f"{rec['id']}: len={len(rec['prompt_token_ids'])} first3={toks[:3]}", flush=True)
    json.dump(out, open("/tmp/hf_a20_fresh.json", "w"))
print("### A_FRESH_DONE", flush=True)
