#!/usr/bin/env python3
"""Teacher-forced ATOM-vs-HF gate.

For each prompt: teacher-force the HF reference's own 64 tokens through HF in one
forward pass to get the top-1 token and top1-top2 logit gap at every step, then
replay the same prefix into a running ATOM service one step at a time and record
its greedy token. Positions are bucketed by the HF margin, so near-ties are
reported separately from positions the oracle can actually decide.
"""
import argparse
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
SHARED = Path("/workspace/ATOM")
REFERENCE = SHARED / "scripts" / "hf_reference_generate.py"
DATASET = Path("/data/datasets/scenario_c_prompts.jsonl")


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hf_step_stats(model, device, prompt_ids, ref_tokens):
    ids = torch.tensor([prompt_ids + ref_tokens], dtype=torch.long, device=device)
    P = len(prompt_ids)
    with torch.no_grad():
        try:
            out = model(input_ids=ids, logits_to_keep=int(ids.shape[1]))
        except TypeError:
            out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logits = logits[0] if logits.dim() == 3 else logits
    rows = []
    for i in range(len(ref_tokens)):
        row = logits[P + i - 1].float()
        top2 = torch.topk(row, 2)
        rows.append({
            "step": i,
            "hf_top1": int(top2.indices[0]),
            "hf_top1_logit": float(top2.values[0]),
            "hf_top2_logit": float(top2.values[1]),
            "hf_gap": float(top2.values[0] - top2.values[1]),
            "ref_token": int(ref_tokens[i]),
        })
    del out, logits
    torch.cuda.empty_cache()
    return rows


def atom_next(url, text, timeout=1800):
    payload = {
        "model": MODEL_PATH,
        "prompt": text,
        "max_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return int(body["token_ids"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    ap.add_argument("--reference", default="/workspace/correctness/hf_scenario_c_first5_eager.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ref = load_reference_module()
    MellumConfig, MellumForCausalLM = ref.load_mellum_classes()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = MellumConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "eager"
    print("loading HF model ...", flush=True)
    model = MellumForCausalLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    ref.load_checkpoint_experts(model, MODEL_PATH)
    model = model.to(args.device).eval()
    print("HF model ready", flush=True)

    records = []
    with open(args.reference, encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= args.limit:
                break
            records.append(json.loads(line))

    results = []
    done_ids = set()
    out_path = Path(args.output)
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            results = prev.get("results", [])
            done_ids = {r["id"] for r in results}
            print(f"resuming: {len(done_ids)} prompts already done", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"could not resume ({exc}); starting fresh", flush=True)
            results = []
            done_ids = set()

    def save():
        summary = save()
    conf_n = summary["confident_positions"]
    conf_m = summary["confident_matched"]
    tie_n = summary["near_tie_positions"]
    tie_m = summary["near_tie_matched"]
    print(
        f"CONFIDENT (HF gap >= 1.0): {conf_m}/{conf_n} = {(conf_m/conf_n*100 if conf_n else 0):.2f}%  |  "
        f"NEAR-TIE (gap < 1.0): {tie_m}/{tie_n} = {(tie_m/tie_n*100 if tie_n else 0):.2f}%  |  "
        f"ALL: {summary['positions_matched']}/{summary['positions_verified']} = {summary['all_rate']*100:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
