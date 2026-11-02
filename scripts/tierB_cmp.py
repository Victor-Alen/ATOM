#!/usr/bin/env python3
"""Tier B: compare production L0/L1 attn & moe outputs against HF."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")
D = Path("/workspace/logs/prodn")


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
    lay = model.model.layers[li]
    lay.self_attn.register_forward_hook(mk(f"L{li}_attn"))
    lay.mlp.register_forward_hook(mk(f"L{li}_moe"))
    lay.register_forward_hook(mk(f"L{li}_blk"))

with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)

for li in range(2):
    for tag in ("attn", "moe", "blk"):
        eng_path = D / f"L{li}_{tag}.pt"
        if not eng_path.exists():
            print(f"L{li}_{tag}: missing", flush=True)
            continue
        e = torch.load(eng_path).float()
        h_all = cap[f"L{li}_{tag}"].float().cpu().reshape(-1, e.shape[-1])
        n = min(e.shape[0], h_all.shape[0])
        e, h = e[:n], h_all[:n]
        d = (e - h).abs()
        per = d.amax(dim=-1)
        bad = (per > 0).nonzero().flatten()
        print(f"L{li}_{tag}: max={d.max().item():.6g} tokens={len(bad)}/{d.shape[0]} "
              f"first={bad[:5].tolist()}", flush=True)
print("### TIERB_DONE", flush=True)


print('--- raw value probe ---', flush=True)
for tag in ('attn',):
    e = torch.load(D / f'L0_{tag}.pt').float()
    h = cap[f'L0_{tag}'].float().cpu().reshape(-1, e.shape[-1])
    print('prod shape', tuple(e.shape), 'hf shape', tuple(h.shape), flush=True)
    print('prod[0,:5]', e[0, :5].tolist(), flush=True)
    print('hf  [0,:5]', h[0, :5].tolist(), flush=True)
    print('prod[1,:5]', e[1, :5].tolist(), flush=True)
    print('hf  [1,:5]', h[1, :5].tolist(), flush=True)
    print('prod abs mean/std', float(e.abs().mean()), float(e.std()), flush=True)
    print('hf   abs mean/std', float(h.abs().mean()), float(h.std()), flush=True)
    best = None
    for sh in range(-3, 4):
        n = min(e.shape[0] - max(sh,0), h.shape[0] - max(-sh,0))
        a = e[max(sh,0):max(sh,0)+n]; b = h[max(-sh,0):max(-sh,0)+n]
        m = float((a-b).abs().max())
        print(f'  shift {sh}: max|d|={m:.4f}', flush=True)
