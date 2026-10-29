#!/usr/bin/env python3
"""Does the first unstable decode step track the *absolute* context length?

If yes, then for a prompt of L tokens the first disagreement should appear
around generated index (65 - L), i.e. always when the running context crosses
the same length. If instead the disagreement always appears at the same
generated index, the threshold is tied to generation steps, not to context.

Prompts are prefixes of one long prompt so the content (and therefore the
density of near-ties) stays comparable across lengths.
"""

import json
import urllib.request
import collections

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
SCEN_C = "/data/datasets/scenario_c_prompts.jsonl"
RUNS = 10
MAX_TOKENS = 96
TARGET_TOKENS = [20, 60, 130, 300, 1000, 3000]


def post(prompt, max_tokens):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.loads(response.read())
    return body["token_ids"], body["usage"]["prompt_tokens"]


def main():
    base = None
    with open(SCEN_C, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == 3:
                base = json.loads(line)["prompt"]
                break
    assert base, "prompt not found"

    # C-0003 is 6183 tokens of this text; scale character length accordingly.
    chars_per_token = len(base) / 6183

    print(f"base text chars={len(base)} (6183 tokens), {chars_per_token:.2f} chars/token")
    print(f"{'target':>7} {'prompt_tok':>10} {'distinct':>9} {'first_idx':>10} {'ctx@first':>10}")
    rows = []
    for target in TARGET_TOKENS:
        n_chars = max(20, int(target * chars_per_token))
        prompt = base[:n_chars]
        streams = []
        prompt_tokens = None
        for _ in range(RUNS):
            ids, prompt_tokens = post(prompt, MAX_TOKENS)
            streams.append(tuple(ids))
        counts = collections.Counter(streams)
        n = len(streams[0])
        first = next((j for j in range(n) if len({s[j] for s in streams}) > 1), None)
        ctx = None if first is None else prompt_tokens + first
        rows.append((target, prompt_tokens, len(counts), first, ctx))
        print(
            f"{target:>7} {prompt_tokens:>10} {len(counts):>9} {str(first):>10} {str(ctx):>10}"
        )

    with open("/workspace/bench/length_sweep.json", "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "target_tokens": t,
                    "prompt_tokens": p,
                    "distinct_streams": d,
                    "first_unstable_index": f,
                    "context_at_first": c,
                }
                for t, p, d, f, c in rows
            ],
            handle,
            indent=2,
        )
    print("wrote /workspace/bench/length_sweep.json")


if __name__ == "__main__":
    main()
