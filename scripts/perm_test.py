#!/usr/bin/env python3
"""Is production L0 attention a permutation of HF's (per row) or a numerical drift?"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")
D = Path("/workspace/logs/prodn33")


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


rec = None
for line in open("/workspace/correctness/hf_scenario_a_20_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] == "A-0006":
        rec = r
        break
ids = torch.tensor([rec["prompt_token_ids"]], dtype=torch.long, device="cuda:0")

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

cap = {}


def mk(name):
    def h(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap.setdefault(name, t.detach())
    return h


for li in range(2):
    L = model.model.layers[li]
    L.self_attn.register_forward_hook(mk(f"attn{li}"))
    L.mlp.register_forward_hook(mk(f"moe{li}"))
    L.register_forward_hook(mk(f"blk{li}"))
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)

for _li in range(2):
  for _tag in ("attn", "moe", "blk"):
    e = torch.load(D / f"L{_li}_{_tag}.pt").float()
    n = int(ids.shape[1])
    e2 = e[:n]
    hf2 = cap[f"{_tag}{_li}"].float().cpu().reshape(n, -1)
    d = (e2 - hf2).abs()
    print(f"L{_li}_{_tag}: max={d.max().item():.6g} mean={d.mean().item():.6g}", flush=True)
print("### PERM_DONE", flush=True)
