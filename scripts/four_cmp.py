#!/usr/bin/env python3
"""Four-tensor semantic comparison for layer 0: attn_out, post-attn residual, moe_out, blk_0.
HF tensors are captured at the same semantic points and the same dtype (bf16)."""
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

cap = {}


def h_layer_input(m, i, o):
    cap["layer_in"] = i[0].detach()


def h_layer_output(m, i, o):
    cap["blk0"] = (o[0] if isinstance(o, tuple) else o).detach()


def h_attn(m, i, o):
    cap["attn0"] = (o[0] if isinstance(o, tuple) else o).detach()


def h_mlp(m, i, o):
    cap["moe0"] = (o[0] if isinstance(o, tuple) else o).detach()


lay0 = model.model.layers[0]
lay0.register_forward_hook(h_layer_input) if False else None
lay0.register_forward_pre_hook(lambda m, a: cap.__setitem__("layer_in", a[0].detach()))
lay0.register_forward_hook(h_layer_output)
lay0.self_attn.register_forward_hook(h_attn)
lay0.mlp.register_forward_hook(h_mlp)

ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)

hf = {
    "attn0_full": cap["attn0"][0],
    "res_after_attn_0": cap["layer_in"][0] + cap["attn0"][0],
    "moe0_full": cap["moe0"][0],
    "blk0_full": cap["blk0"][0],
}
eng = torch.load("/workspace/logs/dumpl7/layers_0.pt")

for name in ("attn0_full", "res_after_attn_0", "moe0_full", "blk0_full"):
    e = eng[name].float().cpu()
    f = hf[name].to(torch.bfloat16).float().cpu()
    if e.shape != f.shape:
        print(f"{name}: SHAPE MISMATCH {tuple(e.shape)} vs {tuple(f.shape)}", flush=True)
        continue
    d = (e - f).abs()
    per_tok = d.amax(dim=-1)
    bad_tok = (per_tok > 0).nonzero().flatten()
    elems = (d > 0).sum().item()
    print(f"{name}: max|delta|={d.max().item():.6g} tokens={len(bad_tok)}/{d.shape[0]} "
          f"elements={elems}/{d.numel()}", flush=True)
    if len(bad_tok):
        print(f"    tokens: {bad_tok[:12].tolist()}", flush=True)
        # which hidden dims differ for the first bad token
        t = int(bad_tok[0])
        dims = (d[t] > 0).nonzero().flatten()
        print(f"    first bad token {t}: dims {dims[:10].tolist()} (n={len(dims)}) "
              f"deltas {d[t][dims[:5]].tolist()}", flush=True)
print("### FOUR_CMP_DONE", flush=True)
