#!/usr/bin/env python3
"""Compare the layer-0 softmax row for token 93 and locate the remaining stage."""
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

ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    out = model(input_ids=ids, logits_to_keep=1, output_attentions=True)
att = out.attentions[0][0]          # [nH, S, S]
print("HF attention shape", tuple(att.shape), flush=True)
row = att[:, 93, :].float().cpu()   # [32, 866]

eng = torch.load("/workspace/logs/dumpl9/layers_0.pt")
pr = eng["pr93"].float()            # [4, 866]
sc = eng["sc93"].float()            # [4, 866]
print("engine pr93", tuple(pr.shape), "sc93", tuple(sc.shape), flush=True)

rep = att.shape[0] // pr.shape[0]
for hk in range(pr.shape[0]):
    hf_row = row[hk * rep]
    d = (pr[hk] - hf_row).abs()
    # also compare against every q head of the group to find the closest match
    best = min(((row[hk * rep + r] - pr[hk]).abs().max().item() for r in range(rep)))
    print(f"kv{hk}: probs max|delta| vs head {hk*rep} = {d.max().item():.6g} (best in group {best:.6g})", flush=True)

# softmax recomputed from the engine's own scores, in bf16 vs fp32
_bf = torch.softmax(sc.double(), dim=-1).float()
print("engine pr vs softmax(engine scores) fp64: ", (pr - _bf).abs().max().item(), flush=True)
print("### ROW_CMP_DONE", flush=True)
