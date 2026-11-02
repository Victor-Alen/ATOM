#!/usr/bin/env python3
"""Valid C measurement: HF recomputed live vs the engine, same prompts, 8 tokens."""
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF = Path("/workspace/ATOM/scripts/hf_reference_generate.py")
IDS = sys.argv[1:] or ["C-0000", "C-0002", "C-0003"]
STEPS = 8


def load():
    spec = importlib.util.spec_from_file_location("hrg", REF)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hrg"] = m
    spec.loader.exec_module(m)
    return m


recs = {}
for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] in IDS:
        recs[r["id"]] = r

ref = load()
Cfg, Lm = ref.load_mellum_classes()
cfg = Cfg.from_pretrained(MODEL)
cfg._attn_implementation = "eager"
model = Lm.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
ref.load_checkpoint_experts(model, MODEL)
model = model.to("cuda:0").eval()
tok = AutoTokenizer.from_pretrained(MODEL)


def hf_tokens(ids_list, steps=STEPS):
    t = torch.tensor([ids_list], dtype=torch.long, device="cuda:0")
    out = []
    with torch.no_grad():
        for _ in range(steps):
            lg = model(input_ids=t, logits_to_keep=1).logits[0, -1]
            nxt = int(lg.argmax())
            out.append(nxt)
            t = torch.cat([t, torch.tensor([[nxt]], device="cuda:0")], dim=1)
    return out


def eng_tokens(text, steps=STEPS, port=8003):
    p = {"model": MODEL, "prompt": text, "max_tokens": steps, "temperature": 0.0,
         "ignore_eos": True, "stream": False}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                data=json.dumps(p).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=7200) as r:
        return json.loads(r.read())["token_ids"]


tot = hit = 0
for pid in IDS:
    rec = recs[pid]
    hf = hf_tokens(rec["prompt_token_ids"])
    eng = eng_tokens(tok.decode(rec["prompt_token_ids"]))
    n = min(len(hf), len(eng))
    m = sum(1 for a, b in zip(eng, hf) if a == b)
    tot += n
    hit += m
    print(f"{pid} (len={len(rec['prompt_token_ids'])}): {m}/{n}  hf={hf[:5]} eng={eng[:5]}", flush=True)
print(f"C-live agreement = {hit}/{tot} = {hit/max(tot,1):.4f}", flush=True)
print("### CLIVE_DONE", flush=True)
