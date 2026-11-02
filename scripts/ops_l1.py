#!/usr/bin/env python3
"""Per-op comparison for layer 1: same input, torch+checkpoint weights vs HF vs engine."""
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")
EPS = 1e-6


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


eng = torch.load("/workspace/logs/dumpl5/layers_0.pt")
h1 = eng["h1_full"]                      # [866, 2304] bf16 (engine post-LN input of L1)
eng_attn = eng["attn_out_1"].float()     # engine attention output (last row)
print("h1_full", tuple(h1.shape), "eng q1_full", tuple(eng["q1_full"].shape) if "q1_full" in eng else "n/a", flush=True)

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()
print("HF ready", flush=True)

modeling = sys.modules["transformers.models.mellum.modeling_mellum"]
cap = {}
state = {"n": 0}
orig_rope = modeling.apply_rotary_pos_emb


def wrap_rope(q, k, cos, sin, unsqueeze_dim=1, *a, **kw):
    cap.setdefault("q_pre_rope", q.detach())
    cap.setdefault("k_pre_rope", k.detach())
    cap.setdefault("cos", cos.detach())
    cap.setdefault("sin", sin.detach())
    out = orig_rope(q, k, cos, sin, unsqueeze_dim, *a, **kw)
    cap.setdefault("q_post_rope", out[0].detach())
    cap.setdefault("k_post_rope", out[1].detach())
    return out


modeling.apply_rotary_pos_emb = wrap_rope
layers = model.model.layers
hl = []
hl.append(layers[1].self_attn.q_proj.register_forward_hook(lambda m, i, o: cap.setdefault("q_lin_hf", o.detach())))
hl.append(layers[1].self_attn.k_proj.register_forward_hook(lambda m, i, o: cap.setdefault("k_lin_hf", o.detach())))
hl.append(layers[1].self_attn.q_norm.register_forward_hook(lambda m, i, o: cap.setdefault("q_norm_hf", o.detach())))
hl.append(layers[1].self_attn.register_forward_hook(
    lambda m, i, o: cap.setdefault("attn_hf", (o[0] if isinstance(o, tuple) else o).detach())))

ids = torch.tensor([json.load(open("/tmp/a0006_ids.json"))], dtype=torch.long, device="cuda:0")
with torch.no_grad():
    model(input_ids=ids, logits_to_keep=1)
modeling.apply_rotary_pos_emb = orig_rope
for h in hl:
    h.remove()
print("captured:", sorted(cap), flush=True)

# manual recomputation from the engine's own input with checkpoint weights
idx = json.load(open(Path(MODEL) / "model.safetensors.index.json"))
wm = idx["weight_map"]
base = "model.layers.1.self_attn."
w = {}
for nm, key in (("q_proj.weight", "wq"), ("k_proj.weight", "wk"), ("v_proj.weight", "wv"),
                ("q_norm.weight", "qn"), ("k_norm.weight", "kn"), ("o_proj.weight", "wo")):
    with safe_open(Path(MODEL) / wm[base + nm], framework="pt") as fh:
        w[key] = fh.get_tensor(base + nm)

h = h1.to("cuda:0")
q_lin = F.linear(h, w["wq"]).float()
q_hf = cap["q_lin_hf"][0].float().reshape(q_lin.shape)
print(f"q_lin   manual vs HF : {(q_lin - q_hf).abs().max().item():.6g}", flush=True)

nH, D = 32, 128
qm = F.linear(h, w["wq"]).view(-1, nH, D)
qn = w["qn"] * (qm.float() * torch.rsqrt(qm.float().pow(2).mean(-1, keepdim=True) + EPS)).to(qm.dtype)
qn_hf = cap["q_norm_hf"][0].float().reshape(-1, nH, D) if cap["q_norm_hf"].dim() == 4 else cap["q_norm_hf"][0].float().reshape(-1, nH, D)
print(f"q_norm  manual vs HF : {(qn.float() - qn_hf).abs().max().item():.6g}", flush=True)

qpre = cap["q_pre_rope"][0].transpose(0, 1).float()  # [S, nH, D]
print(f"q_pre_rope manual vs HF: {(qn.float() - qpre).abs().max().item():.6g}", flush=True)
print("### OPS_DONE", flush=True)
