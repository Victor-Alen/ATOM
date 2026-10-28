#!/usr/bin/env python3
"""Measure how close to a tie the HF reference itself is at every decode step.

Motivation: on the scenario-C prompts the ATOM-vs-HF greedy agreement is only
~60%, yet HF eager and HF SDPA also disagree with each other.  Before blaming a
structural bug in the ATOM port we need to know whether the reference greedy path
runs through near-ties: if top1-top2 logit margins are tiny, bf16 kernel
differences alone flip the argmax and per-token agreement cannot be stable.

For each prompt we teacher-force HF's own 64 reference tokens through the model
in one forward pass and record the top-1/top-2 logit margin at every step, then
join that with the ATOM divergence positions.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

MODEL_PATH = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPT = REPO_ROOT / "scripts" / "hf_reference_generate.py"


def load_reference_module():
    spec = importlib.util.spec_from_file_location("hf_reference_generate", REFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-reference", default=str(REPO_ROOT / "correctness" / "hf_scenario_c_200_eager.jsonl"))
    parser.add_argument("--atom-compare", default=str(REPO_ROOT / "correctness" / "scenario_c_first5_windowfix_vs_eager" / "per_prompt.json"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    ref = load_reference_module()
    MellumConfig, MellumForCausalLM = ref.load_mellum_classes()

    references = {}
    with open(args.hf_reference, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if len(references) >= args.limit:
                break
            references[record["id"]] = record

    atom = {}
    if Path(args.atom_compare).exists():
        for record in json.loads(Path(args.atom_compare).read_text(encoding="utf-8")):
            atom[record["id"]] = record

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = MellumConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "eager"
    dtype = torch.bfloat16
    print(f"Loading HF model on {args.device} ({dtype}) ...", flush=True)
    model = MellumForCausalLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    ref.load_checkpoint_experts(model, MODEL_PATH)
    model = model.to(args.device).eval()
    print("Model ready.", flush=True)

    results = []
    for prompt_id, record in references.items():
        prompt = record["prompt"]
        hf_ids = record["token_ids"]
        if record.get("prompt_token_ids"):
            input_ids = torch.tensor([record["prompt_token_ids"]], dtype=torch.long)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prefix_len = input_ids.shape[1]
        # Teacher force: feed prefix + all but the last reference token so that the
        # logits at step i predict reference token i.
        fed = torch.cat(
            [input_ids[0], torch.tensor(hf_ids[:-1], dtype=torch.long)]
        ).unsqueeze(0).to(args.device)

        with torch.no_grad():
            # logits_to_keep == 0 in this modeling file means "last token only",
            # so ask explicitly for every position.
            keep = fed.shape[1]
            logits = model(input_ids=fed, use_cache=False, logits_to_keep=keep).logits[0].float()
            logits_again = model(input_ids=fed, use_cache=False, logits_to_keep=keep).logits[0].float()

        steps = []
        deterministic = True
        for i, expected in enumerate(hf_ids):
            row = logits[prefix_len - 1 + i]
            row2 = logits_again[prefix_len - 1 + i]
            top2 = torch.topk(row, 2)
            argmax = int(top2.indices[0].item())
            gap = float((top2.values[0] - top2.values[1]).item())
            prob = float(torch.softmax(row, dim=-1)[argmax].item())
            if argmax != int(row2.argmax().item()):
                deterministic = False
            steps.append(
                {
                    "step": i,
                    "top1": argmax,
                    "expected": int(expected),
                    "argmax_matches_reference": argmax == int(expected),
                    "gap": gap,
                    "top1_prob": prob,
                }
            )

        mismatched = [s["step"] for s in steps if not s["argmax_matches_reference"]]
        atom_record = atom.get(prompt_id)
        atom_first_divergence = None
        if atom_record:
            for i, (a, h) in enumerate(zip(atom_record["atom_token_ids"], atom_record["hf_token_ids"])):
                if a != h:
                    atom_first_divergence = i
                    break

        gaps = sorted(s["gap"] for s in steps)
        results.append(
            {
                "id": prompt_id,
                "prompt_tokens": prefix_len,
                "self_deterministic": deterministic,
                "reference_self_consistent": not mismatched,
                "reference_self_mismatch_steps": mismatched,
                "atom_first_divergence": atom_first_divergence,
                "gap_at_atom_divergence": steps[atom_first_divergence]["gap"] if atom_first_divergence is not None else None,
                "prob_at_atom_divergence": steps[atom_first_divergence]["top1_prob"] if atom_first_divergence is not None else None,
                "gap_min": gaps[0],
                "gap_p50": gaps[len(gaps) // 2],
                "gap_max": gaps[-1],
                "steps_with_gap_lt_0.25": sum(1 for g in gaps if g < 0.25),
                "steps_with_gap_lt_1.0": sum(1 for g in gaps if g < 1.0),
                "steps": steps,
            }
        )
        print(
            f"{prompt_id}: len={prefix_len} self_ok={not mismatched} det={deterministic} "
            f"atom_first_div={atom_first_divergence} gap@div={results[-1]['gap_at_atom_divergence']} "
            f"gap_p50={results[-1]['gap_p50']:.4f} n<0.25={results[-1]['steps_with_gap_lt_0.25']}/64",
            flush=True,
        )

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
