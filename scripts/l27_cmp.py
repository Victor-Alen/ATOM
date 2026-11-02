#!/usr/bin/env python3
"""At 1200 tokens, isolate whether the difference is L27 itself, the final norm, or lm_head."""
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

cap = {}
lay27 = model.model.layers[27]
lay27.register_forward_pre_hook(lambda m, a: cap.__setitem__("l27_in", a[0].detach()))
lay27.register_forward_hook(lambda m, i, o: cap.__setitem__("l27_out", (o[0] if isinstance(o, tuple) else o).detach()))
model.model.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("norm_in", a[0].detach()))

with torch.no_grad():
    out = model(input_ids=ids, logits_to_keep=1)

eng = torch.load("/workspace/logs/dumpc/layers_0.pt")
def cmp(name, e, t):
    d = (e.float().cpu() - t.float().cpu()).abs().max().item()
    print(f"{name}: {d:.6g}", flush=True)

cmp("L26 out (engine blk_26) vs HF l27_in", eng["blk_26"], cap["l27_in"][0, -1])
cmp("blk_27 vs HF norm_in", eng["blk_27"], cap["norm_in"][0, -1])
print("hf top1:", int(out.logits[0, -1].argmax()), flush=True)
print("### L27_CMP_DONE", flush=True)
