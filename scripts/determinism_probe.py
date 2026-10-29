#!/usr/bin/env python3
"""Measure the greedy nondeterminism with a statistic that can actually judge a fix.

With 8 repeats the distinct-stream count is too coarse: a step that flips rarely
shows 1 stream most of the time and 2-3 occasionally, so one lucky draw can make
any configuration look fixed. This probe repeats each prompt N times (default 40)
and reports:

  modal_rate  = (count of the most frequent output) / N
  distinct    = how many different outputs appeared
  prefix      = common token prefix shared by ALL runs

modal_rate is the number to compare across configurations: it is a proportion, so
its sampling noise is bounded by the binomial standard error sqrt(p(1-p)/N),
which at N=40 and p=0.5 is about 0.08. Two configurations whose modal rates differ
by less than ~2 standard errors are not distinguishable.
"""

import argparse
import collections
import json
import math
import urllib.request

MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
SCEN_C = "/data/datasets/scenario_c_prompts.jsonl"


def post(url, prompt, max_tokens):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())["token_ids"]


def load_prompts(include_short):
    prompts = []
    if include_short:
        prompts.append(
            {"id": "short64", "prompt": "def add(a, b):\n    return", "max_tokens": 64}
        )
    with open(SCEN_C, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in (0, 3):
                prompts.append(
                    {"id": f"C-{index:04d}", "prompt": json.loads(line)["prompt"], "max_tokens": 64}
                )
            if index > 3:
                break
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-short", action="store_true")
    args = parser.parse_args()

    report = {"tag": args.tag, "repeat": args.repeat, "prompts": []}
    for item in load_prompts(args.include_short):
        streams = [tuple(post(args.url, item["prompt"], item["max_tokens"]))
                   for _ in range(args.repeat)]
        counts = collections.Counter(streams)
        modal, modal_count = counts.most_common(1)[0]
        n = len(streams[0])
        prefix = next((j for j in range(n) if len({s[j] for s in streams}) > 1), n)
        modal_rate = modal_count / args.repeat
        se = math.sqrt(modal_rate * (1 - modal_rate) / args.repeat)
        row = {
            "id": item["id"],
            "repeat": args.repeat,
            "distinct": len(counts),
            "modal_rate": modal_rate,
            "modal_stderr": se,
            "modal_count": modal_count,
            "all_counts": sorted(counts.values(), reverse=True),
            "common_prefix": prefix,
            "output_len": n,
        }
        report["prompts"].append(row)
        print(
            f"{item['id']:>9}: distinct={row['distinct']:>2}  "
            f"modal_rate={modal_rate:.3f} (+-{se:.3f})  "
            f"counts={row['all_counts']}  common_prefix={prefix}/{n}",
            flush=True,
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
