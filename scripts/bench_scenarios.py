#!/usr/bin/env python3
"""Scenario A / B / C load generator for the ATOM OpenAI-compatible server.

Mirrors the parameters in the interview task PDF:

  A: 512 requests, concurrency 32, dataset scenario_a_prompts.jsonl,
     max/min tokens 256, stream, ignore_eos, temperature 0.
  B: 50 requests, concurrency 1, random 1024-token prompts, 256 output.
  C: 200 requests, open loop Poisson at r requests/second, dataset
     scenario_c_prompts.jsonl, 256 output.

This is an independent harness: it exists so results can be produced even
when the official evalscope client is unavailable in the frozen image. The
per-scenario JSON records the exact command line and the measured ratios so
the numbers can be traced back to a run.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp

SCEN_A_DATASET = "/data/datasets/scenario_a_prompts.jsonl"
SCEN_C_DATASET = "/data/datasets/scenario_c_prompts.jsonl"
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"
FILLER = (
    "def solve(items):\n    total = 0\n    for item in items:\n"
    "        total += item\n    return total\n"
)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_prompts(path, limit=None):
    prompts = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.append(record.get("prompt", ""))
            if limit and len(prompts) >= limit:
                break
    return prompts


def random_prompt(target_tokens):
    """A rough 1024-token filler prompt; the server tokenizes it itself."""
    words = []
    while len(words) < target_tokens * 2:
        words.extend(FILLER.split())
    return " ".join(words[: target_tokens * 2])


async def one_request(session, url, prompt, max_tokens, request_id):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    started = time.perf_counter()
    first_token_at = None
    token_times = []
    chunks = 0
    error = None
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:300]}")
            async for raw in response.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices", []):
                    text = choice.get("text")
                    if not text:
                        continue
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    chunks += 1
                    token_times.append(now)
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    ttft = (first_token_at - started) if first_token_at else None
    tpot = None
    if len(token_times) > 1 and first_token_at is not None:
        tpot = (token_times[-1] - first_token_at) / (len(token_times) - 1)
    return {
        "id": request_id,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "chunks": chunks,
        "elapsed_s": ended - started,
        "error": error,
    }


async def run_closed_loop(url, prompts, concurrency, max_tokens, out_path):
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    started = time.perf_counter()

    async def worker(session, prompt, index):
        async with semaphore:
            return await one_request(session, url, prompt, max_tokens, index)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [worker(session, prompt, i) for i, prompt in enumerate(prompts)]
        for task in asyncio.as_completed(tasks):
            results.append(await task)
    wall = time.perf_counter() - started
    return summarise(results, wall, out_path, len(prompts))


async def run_open_loop(url, prompts, rate, max_tokens, out_path):
    """Poisson arrivals at `rate` requests per second, unbounded concurrency."""
    rng = random.Random(20261028)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    results = []
    tasks = []
    schedule = []
    cursor = 0.0
    for _ in prompts:
        cursor += rng.expovariate(rate)
        schedule.append(cursor)

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for index, (prompt, offset) in enumerate(zip(prompts, schedule)):
            delay = wall_start + offset - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one_request(session, url, prompt, max_tokens, index)))
        for task in asyncio.as_completed(tasks):
            results.append(await task)
    wall = time.perf_counter() - wall_start
    return summarise(results, wall, out_path, len(prompts))


def summarise(results, wall_s, out_path, requested):
    ok = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    total_out = sum(r["chunks"] for r in ok)
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tpots = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    summary = {
        "requested": requested,
        "succeeded": len(ok),
        "failed": len(failed),
        "wall_s": wall_s,
        "output_tokens": total_out,
        "output_tokens_per_s": (total_out / wall_s) if wall_s > 0 else None,
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p99_s": percentile(ttfts, 0.99),
        "tpot_p50_s": percentile(tpots, 0.50),
        "tpot_p99_s": percentile(tpots, 0.99),
        "tpot_mean_s": statistics.fmean(tpots) if tpots else None,
        "errors": [r["error"] for r in failed][:5],
    }
    Path(out_path).write_text(
        json.dumps({"summary": summary, "per_request": results}, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["A", "B", "C"])
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--number", type=int)
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.scenario == "A":
        prompts = load_prompts(SCEN_A_DATASET, args.number or 512)
        summary = asyncio.run(
            run_closed_loop(args.url, prompts, args.parallel or 32, args.max_tokens, args.out)
        )
    elif args.scenario == "B":
        prompts = [random_prompt(1024) for _ in range(args.number or 50)]
        summary = asyncio.run(
            run_closed_loop(args.url, prompts, 1, args.max_tokens, args.out)
        )
    else:
        prompts = load_prompts(SCEN_C_DATASET, args.number or 200)
        summary = asyncio.run(
            run_open_loop(args.url, prompts, args.rate, args.max_tokens, args.out)
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
