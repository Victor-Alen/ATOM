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
    for rec in records:
        pid = rec["id"]
        prompt_ids = rec["prompt_token_ids"][: 64] if False else rec["prompt_token_ids"]
        ref_tokens = rec["token_ids"][: args.steps]
        rows = hf_step_stats(model, args.device, prompt_ids, ref_tokens)
        matched = verified = skipped = 0
        for row in rows:
            t = row["step"]
            prefix_ids = prompt_ids + ref_tokens[:t]
            text = tokenizer.decode(prefix_ids)
            back = tokenizer.encode(text, add_special_tokens=False)
            if back != prefix_ids:
                skipped += 1
                row["atom"] = None
                row["skip_reason"] = "tokenizer_roundtrip"
                continue
            verified += 1
            got = atom_next(args.url, text)
            row["atom"] = got
            if got == row["hf_top1"]:
                matched += 1
                row["match"] = True
            else:
                row["match"] = False
        buckets = {}
        for lo in (1.0,):
            sel = [r for r in rows if r.get("atom") is not None and r["hf_gap"] >= lo]
            buckets[f"gap_ge_{lo}"] = {
                "n": len(sel),
                "matched": sum(1 for r in sel if r["match"]),
            }
        sel_ties = [r for r in rows if r.get("atom") is not None and r["hf_gap"] < 1.0]
        buckets["gap_lt_1.0"] = {
            "n": len(sel_ties),
            "matched": sum(1 for r in sel_ties if r["match"]),
        }
        entry = {
            "id": pid,
            "prompt_tokens": len(prompt_ids),
            "steps": len(rows),
            "verified": verified,
            "skipped": skipped,
            "matched_all": matched,
            "all_rate": matched / verified if verified else 0.0,
            "buckets": buckets,
            "rows": rows,
        }
        results.append(entry)
        print(
            f"{pid}: verified {verified}/{len(rows)} matched {matched} "
            f"({entry['all_rate']*100:.1f}%) gap>=1.0 {buckets['gap_ge_1.0']['matched']}/{buckets['gap_ge_1.0']['n']} "
            f"gap<1.0 {buckets['gap_lt_1.0']['matched']}/{buckets['gap_lt_1.0']['n']}",
            flush=True,
        )

    conf_n = sum(e["buckets"]["gap_ge_1.0"]["n"] for e in results)
    conf_m = sum(e["buckets"]["gap_ge_1.0"]["matched"] for e in results)
    tie_n = sum(e["buckets"]["gap_lt_1.0"]["n"] for e in results)
    tie_m = sum(e["buckets"]["gap_lt_1.0"]["matched"] for e in results)
    summary = {
        "variants": str(args.url),
        "prompts": len(results),
        "confident_positions": conf_n,
        "confident_matched": conf_m,
        "confident_rate": conf_m / conf_n if conf_n else 0.0,
        "near_tie_positions": tie_n,
        "near_tie_matched": tie_m,
        "near_tie_rate": tie_m / tie_n if tie_n else 0.0,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=1))
    print(
        f"CONFIDENT (HF gap >= 1.0): {conf_m}/{conf_n} = {(conf_m/conf_n*100 if conf_n else 0):.2f}%  |  "
        f"NEAR-TIE (gap < 1.0): {tie_m}/{tie_n} = {(tie_m/tie_n*100 if tie_n else 0):.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
