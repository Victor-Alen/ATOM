#!/usr/bin/env python3
"""Compare ATOM and saved HF greedy outputs token by token."""

import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF reference JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="/data/models/Mellum2.1-12B-A2.5B-Thinking")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt", help="Prompt override for a single smoke record")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    if args.limit is not None:
        records = records[: args.limit]

    results = []
    with (output_dir / "atom_outputs.jsonl").open("w", encoding="utf-8") as saved:
        for index, reference in enumerate(records):
            prompt = args.prompt if args.prompt is not None else reference["prompt"]
            request = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": False,
            }
            http = urllib.request.Request(
                args.url,
                data=json.dumps(request).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(http, timeout=600) as response:
                atom = json.load(response)
            saved.write(json.dumps({"id": reference["id"], "response": atom}, ensure_ascii=False) + "\n")
            saved.flush()

            atom_ids = atom.get("token_ids")
            if atom_ids is None:
                raise RuntimeError("ATOM response has no token_ids; restart the server with the diagnostic response change")
            reference_ids = reference["token_ids"][: args.max_tokens]
            atom_ids = atom_ids[: args.max_tokens]
            count = min(len(reference_ids), len(atom_ids))
            if count != args.max_tokens:
                raise RuntimeError(f"{reference['id']}: expected {args.max_tokens} tokens, got HF={len(reference_ids)} ATOM={len(atom_ids)}")
            matches = sum(a == b for a, b in zip(reference_ids, atom_ids))
            first_mismatch = next((i for i, (a, b) in enumerate(zip(reference_ids, atom_ids)) if a != b), None)
            result = {
                "id": reference["id"],
                "prompt_tokens": reference["prompt_tokens"],
                "matches": matches,
                "total": count,
                "agreement": matches / count,
                "exact_sequence": matches == count,
                "first_mismatch": first_mismatch,
                "hf_token_ids": reference_ids,
                "atom_token_ids": atom_ids,
            }
            results.append(result)
            print(f"{index + 1}/{len(records)} {result['id']}: {matches}/{count} first_mismatch={first_mismatch}", flush=True)

    total = sum(item["total"] for item in results)
    matches = sum(item["matches"] for item in results)
    summary = {
        "count": len(results),
        "tokens_per_prompt": args.max_tokens,
        "matched_positions": matches,
        "total_positions": total,
        "token_position_agreement": matches / total if total else 0.0,
        "exact_sequence_count": sum(item["exact_sequence"] for item in results),
        "exact_sequence_ratio": sum(item["exact_sequence"] for item in results) / len(results) if results else 0.0,
        "threshold": 0.90,
        "token_position_gate_passed": total > 0 and matches / total >= 0.90,
    }
    (output_dir / "per_prompt.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
