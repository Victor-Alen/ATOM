#!/usr/bin/env python3
"""Record greedy token IDs from the running server for a fixed prompt set.

Used as the ablation baseline: after changing the server configuration the same
prompts must produce the same token IDs for the change to count as
numerics-preserving.
"""

import argparse
import json
import urllib.request

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
SCEN_C = "/data/datasets/scenario_c_prompts.jsonl"


def post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    prompts = [{"id": "short", "prompt": "def add(a, b):\n    return"}]
    with open(SCEN_C, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in (0, 3):
                record = json.loads(line)
                prompts.append({"id": record["id"], "prompt": record["prompt"]})
            if index > 3:
                break

    results = []
    for item in prompts:
        body = post(
            args.url,
            {
                "model": MODEL,
                "prompt": item["prompt"],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": False,
            },
        )
        results.append(
            {
                "id": item["id"],
                "prompt_chars": len(item["prompt"]),
                "usage": body.get("usage"),
                "token_ids": body.get("token_ids"),
            }
        )
        print(f"{item['id']}: {body.get('usage')} tokens={body.get('token_ids')}", flush=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
