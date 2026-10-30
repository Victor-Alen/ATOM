#!/usr/bin/env python3
"""Generate HF eager greedy references (64 tokens) for scenario A / B prompt sets."""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

SHARED = Path("/workspace/ATOM")
REFERENCE = SHARED / "scripts" / "hf_reference_generate.py"
MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
FILLER = (
    "def solve(items):\n    total = 0\n    for item in items:\n"
    "        total += item\n    return total\n"
)


def load_reference_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario_prompts(kind, limit, tokenizer):
    if kind == "A":
        out = []
        with open("/data/datasets/scenario_a_prompts.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if len(out) >= limit:
                    break
                rec = json.loads(line)
                out.append((rec["id"], rec["prompt"]))
        return out
    if kind == "C":
        out = []
        with open("/data/datasets/scenario_c_prompts.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if len(out) >= limit:
                    break
                rec = json.loads(line)
                out.append((rec["id"], rec["prompt"]))
        return out
    if kind == "B":
        # ~1024 token prompts built from the same filler the benchmark uses
        out = []
        rng = random.Random(20261030)
        for i in range(limit):
            parts = [f"# case {i} rng {rng.randrange(10**6)}\n"]
            n_tok = 0
            while n_tok < 1024:
                parts.append(FILLER)
                n_tok += len(tokenizer(FILLER, add_special_tokens=False)["input_ids"])
            out.append((f"B-{i:04d}", "".join(parts)))
        return out
    raise SystemExit(f"unknown scenario {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--output", required=True)
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
    print("HF ready", flush=True)

    prompts = scenario_prompts(args.scenario, args.limit, tokenizer)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            done.add(json.loads(line)["id"])
        print(f"resume: {len(done)} done", flush=True)

    with out.open("a", encoding="utf-8") as fh:
        for pid, text in prompts:
            if pid in done:
                continue
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids,
                    max_new_tokens=args.steps,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = gen[0, ids.shape[1]:].tolist()
            rec = {
                "id": pid,
                "prompt": text,
                "prompt_token_ids": ids[0].tolist(),
                "token_ids": new_tokens,
                "prompt_tokens": int(ids.shape[1]),
                "attention": "eager",
                "dtype": "torch.bfloat16",
            }
            fh.write(json.dumps(rec) + chr(10))
            fh.flush()
            print(f"{pid}: prompt_tokens={ids.shape[1]} gen={len(new_tokens)} head={new_tokens[:5]}", flush=True)
    print("### DONE", flush=True)


if __name__ == "__main__":
    main()
