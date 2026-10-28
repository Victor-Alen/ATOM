# Measured results

Machine: the assigned node, 2x AMD Instinct MI308X, frozen image.
Model: `/data/models/Mellum2.1-12B-A2.5B-Thinking`.
All numbers below are single card. Client: `scripts/bench_scenarios.py` (an
independent harness, because `evalscope` is not installed in the frozen image).
Client caveat: it reads the `prompt` field of each JSONL record, whereas
evalscope's `line_by_line` reader treats the whole line as the prompt text, so
absolute numbers are not bit-comparable with the published baseline; the
level-0 vs level-3 rows were produced back to back with the same client on the
same machine and *are* directly comparable.

## Step 0 configuration

```
python3 -m atom.entrypoints.openai_server \
  --model /data/models/Mellum2.1-12B-A2.5B-Thinking \
  --level 0 --enforce-eager --kv_cache_dtype bf16 --max-model-len 32768 \
  --gpu-memory-utilization 0.35 --host 127.0.0.1 --server-port 8000 \
  --no-enable_prefix_caching
```
Compile level 0, eager, bf16 KV, MTP off, SWA on, prefix caching off - i.e. the
task's "most naive port".

## Scenario A - throughput, 512 requests, concurrency 32, 256 output, ignore_eos

| run | succeeded | wall (s) | tokens/s/GPU | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|-----|-----------|----------|--------------|----------|----------|----------|----------|
| published baseline (1 GPU) | - | 205.35 | 638.28 | - | 5.010 s | 38.95 ms | 50.14 ms |
| Step 0 (level 0, eager) | 512/512 | 183.54 | **714.09** | 2.177 s | 3.472 s | 36.68 ms | 45.29 ms |
| + point 1 (level 3 + cudagraph) | 512/512 | 130.69 | **1002.90** | 2.661 s | 3.384 s | 22.27 ms | 31.65 ms |

## Scenario B - single stream, 50 requests, concurrency 1, ~1k input, 256 output

| run | succeeded | wall (s) | TTFT p99 | TPOT p50 | TPOT p99 |
|-----|-----------|----------|----------|----------|----------|
| Step 0 (level 0, eager) | 50/50 | 378.26 | 0.157 s | 29.06 ms | 29.32 ms |
| + point 1 (level 3 + cudagraph) | 50/50 | 69.87 | 0.159 s | **4.89 ms** | 4.90 ms |

## Scenario C - Poisson arrivals r=0.30/s, 200 requests, 256 output

| run | succeeded | wall (s) | tokens/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|-----|-----------|----------|----------|----------|----------|----------|----------|
| published baseline (1 GPU) | - | 671.77 | 76.22 | - | 0.994 s | 36.29 ms | 41.27 ms |
| Step 0 (level 0, eager) | 200/200 | 633.24 | 80.85 | 0.480 s | 1.141 s | 34.12 ms | 42.45 ms |
| + point 1 | not measured yet | | | | | | |

## Point 1 correctness status (not certified)

Same three prompts through both servers, greedy, `ignore_eos`, 32 tokens:

| prompt | input tokens | level 0 vs level 3 token match |
|--------|--------------|-------------------------------|
| `def add(a, b): return` | 10 | 32/32 identical |
| C-0000 | 7881 | 32/32 identical |
| C-0003 | 6183 | **2/32** - diverges at step 2 |

So level 3 is a large win (1.40x scenario A throughput, 5.9x single-stream TPOT)
but it is **not yet numerics-preserving**. Level 3 splits the graph at
`aiter.unified_attention_with_output`, which routes prefill through
`forward_impl_server_mode` instead of the varlen FlashAttention call that
`5b72cc44` fixed. The sliding-window convention is not uniform in that file:

- decode: `(self.sliding_window - 1, 0)`
- varlen FlashAttention prefill: `(self.sliding_window - 1, 0, 0)` (fixed in `5b72cc44`)
- paged/persistent-ASM path: `sliding_window=self.sliding_window` - raw, unadjusted

One of those is wrong by one key. Confirming which requires reading the ASM
kernel's window convention; until then point 1 cannot be claimed as passing the
correctness gate.
