#!/usr/bin/env python3
"""Golden per-layer fingerprints for prompt + one generated token (7882 rows).

Uses HF only, so it can be compared against the in-engine reference path.
"""
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REFERENCE = Path("/workspace/ATOM/scripts/hf_reference_generate.py")


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=0)
    args_in = ap.parse_args()
    ref = load_reference_module()
    MellumConfig, MellumForCausalLM = ref.load_mellum_classes()
    cfg = MellumConfig.from_pretrained(MODEL_PATH)
    cfg._attn_implementation = "eager"
    print("loading HF ...", flush=True)
    m = MellumForCausalLM.from_pretrained(MODEL_PATH, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    ref.load_checkpoint_experts(m, MODEL_PATH)
    m = m.to("cuda:0").eval()
    print("HF ready", flush=True)

    rec = None
    for line in open("/workspace/correctness/hf_scenario_c_200_eager.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["id"] == "C-0000":
            rec = r
            break
    seq = list(rec["prompt_token_ids"])
    if args_in.tokens:
        seq = seq[: args_in.tokens]
    if args_in.extra:
        seq = seq + list(rec["token_ids"][: args_in.extra])
    ids = torch.tensor([seq], dtype=torch.long, device="cuda:0")
    print("rows:", ids.shape[1], flush=True)

    vals = []
    handles = []
    for i, layer in enumerate(m.model.layers):
        def mk(i):
            def fn(mod, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                vals.append((i, float(t[0, -1].float().abs().max())))
            return fn
        handles.append(layer.register_forward_hook(mk(i)))
    with torch.no_grad():
        out = m(input_ids=ids, logits_to_keep=1)
    for h in handles:
        h.remove()
    print("layer absmax (last row):", [round(v, 3) for _, v in vals][:8], flush=True)
    top = out.logits[0, -1].float().topk(3)
    print("hf top3:", top.indices.tolist(), [round(x, 4) for x in top.values.tolist()], flush=True)
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
