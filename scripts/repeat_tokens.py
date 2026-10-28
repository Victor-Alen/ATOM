#!/usr/bin/env python3
"""Send the same greedy request repeatedly and count distinct token streams.

If the engine is deterministic, every repetition returns the same IDs. A
non-deterministic MoE reduction (e.g. atomic accumulation) or a KV-cache race
shows up here as several distinct streams for one prompt.
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
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--include-short",
        action="store_true",
        help="also probe a 10-token prompt with 64 output tokens: if a long "
        "context is unstable but this is stable, the nondeterminism lives in "
        "attention/KV rather than in the MoE reduction",
    )
    args = parser.parse_args()

    prompts = []
    if args.include_short:
        prompts.append(
            {"id": "short64", "prompt": "def add(a, b):\n    return", "max_tokens": 64}
        )
    with open(SCEN_C, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in (0, 3):
                record = json.loads(line)
                prompts.append(
                    {"id": record["id"], "prompt": record["prompt"], "max_tokens": args.max_tokens}
                )
            if index > 3:
                break

    report = {"tag": args.tag, "repeat": args.repeat, "prompts": []}
    for item in prompts:
        streams = []
        for _ in range(args.repeat):
            body = post(
                args.url,
                {
                    "model": MODEL,
                    "prompt": item["prompt"],
                    "max_tokens": item["max_tokens"],
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "stream": False,
                },
            )
            streams.append(tuple(body["token_ids"]))
        distinct = {s: streams.count(s) for s in set(streams)}
        report["prompts"].append(
            {
                "id": item["id"],
                "distinct_streams": len(distinct),
                "counts": sorted(distinct.values(), reverse=True),
                "streams": [list(s) for s in distinct],
            }
        )
        print(
            f"{args.tag} {item['id']}: {len(distinct)} distinct stream(s) over "
            f"{args.repeat} runs, counts={sorted(distinct.values(), reverse=True)}",
            flush=True,
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
