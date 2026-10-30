#!/usr/bin/env python3
"""Recompute the scenario overlap rates from the raw ATOM outputs and HF references.

Deliberately does not read summary.json: the numbers here are recomputed from the
per-prompt token ids so they can be checked independently.
"""
import json
import sys
from pathlib import Path

BASE = Path("/workspace/correctness")

CASES = [
    ("A (20 prompts)", "scen_A_triton", "hf_scenario_a_20_eager.jsonl"),
    ("B (20 prompts)", "scen_B_triton", "hf_scenario_b_20_eager.jsonl"),
    ("C (200 prompts)", "scen_C_triton", "hf_scenario_c_200_eager.jsonl"),
    ("C v23 (20 prompts)", "scen_C_v23", "hf_scenario_c_200_eager.jsonl"),
]


def load_hf(name):
    refs = {}
    for line in (BASE / name).open(encoding="utf-8"):
        rec = json.loads(line)
        refs[rec["id"]] = rec["token_ids"]
    return refs


def load_atom(dirname):
    out = {}
    for line in (BASE / dirname / "atom_outputs.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        ids = rec["response"]["choices"][0].get("token_ids")
        if ids is None:
            ids = rec["response"].get("token_ids")
        out[rec["id"]] = list(ids) if ids is not None else []
    return out


def main():
    print(f"{'scenario':20s} {'prompts':>7s} {'positions':>10s} {'agree':>7s} {'exact seqs':>10s}")
    for label, dirname, hfname in CASES:
        if not (BASE / dirname / "atom_outputs.jsonl").exists():
            print(f"{label:20s} {'MISSING':>7s}")
            continue
        hf = load_hf(hfname)
        atom = load_atom(dirname)
        matched = total = exact = used = 0
        for pid, ids in atom.items():
            ref = hf.get(pid)
            if ref is None:
                continue
            n = min(len(ids), 64, len(ref))
            if n == 0:
                continue
            used += 1
            same = sum(1 for i in range(n) if ids[i] == ref[i])
            matched += same
            total += n
            if same == n:
                exact += 1
        rate = matched / total if total else 0.0
        print(f"{label:20s} {used:7d} {total:10d} {rate:7.4f} {exact:5d}/{used:<4d}")
    print()
    print("note: 'agree' counts token positions over the first 64 generated tokens;")
    print("      'exact seqs' counts prompts whose first 64 tokens all match.")


if __name__ == "__main__":
    main()
