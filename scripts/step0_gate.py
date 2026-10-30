#!/usr/bin/env python3
"""Compare the engine's greedy tokens with HF references for Step 0.

Usage:
    python3 scripts/step0_gate.py --port 8001 --prompt-id C-0000 --tokens 1024 --max-tokens 8

The expected HF tokens are read from a JSON file written by the A100 reference run
(\`/tmp/mellum/hf_1024_tokens.json\`) or from the full-prompt reference JSONL.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
REF_JSONL = "/workspace/correctness/hf_scenario_c_200_eager.jsonl"


def load_prompt(prompt_id: str) -> dict:
    with open(REF_JSONL, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["id"] == prompt_id:
                return rec
    raise SystemExit(f"prompt {prompt_id} not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--prompt-id", default="C-0000")
    ap.add_argument("--tokens", type=int, default=1024, help="how many prompt tokens to send")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--hf-tokens", default="/tmp/mellum/hf_1024_tokens.json",
                    help="optional JSON with the expected HF greedy tokens for this prefix")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    rec = load_prompt(args.prompt_id)
    ids = rec["prompt_token_ids"][: args.tokens]
    text = tok.decode(ids)
    got_ids = tok.encode(text, add_special_tokens=False)
    if got_ids != ids:
        raise SystemExit("tokenizer round-trip failed; refusing to compare")

    payload = {
        "model": MODEL,
        "prompt": text,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=7200) as resp:
        atom = json.loads(resp.read())["token_ids"]

    if args.hf_tokens and args.tokens == 1024:
        try:
            hf = json.load(open(args.hf_tokens))[: args.max_tokens]
        except FileNotFoundError:
            hf = rec["token_ids"][: args.max_tokens]
    else:
        hf = rec["token_ids"][: args.max_tokens]

    print(f"prompt_tokens={len(ids)} steps={args.max_tokens}")
    print("atom:", atom)
    print("hf  :", hf)
    print("match:", atom == hf)
    if atom != hf:
        same = sum(1 for a, b in zip(atom, hf) if a == b)
        print(f"prefix agreement: {same}/{len(hf)}")


if __name__ == "__main__":
    main()
