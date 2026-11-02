#!/usr/bin/env python3
"""Isolate whether divergence starts when the sliding window engages (>1024 tokens)."""
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
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
ids = torch.tensor([rec["prompt_token_ids"][:n]], dtype=torch.long, device="cuda:0")
print("using tokens:", n, flush=True)

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()

out = []
with torch.no_grad():
    for _ in range(8):
        lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        nxt = int(lg.argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda:0")], dim=1)
print("hf_tokens:", out, flush=True)

# engine
import urllib.request  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
text = tok.decode(rec["prompt_token_ids"][:n])
p = {"model": MODEL, "prompt": text, "max_tokens": 8, "temperature": 0.0,
     "ignore_eos": True, "stream": False}
r = urllib.request.Request("http://127.0.0.1:8001/v1/completions", data=json.dumps(p).encode(),
                           headers={"Content-Type": "application/json"})
eng = json.loads(urllib.request.urlopen(r, timeout=3600).read())["token_ids"]
print("engine   :", eng, flush=True)
print("match:", eng[:8] == out, flush=True)
print("### WINDOW_TEST_DONE", flush=True)
