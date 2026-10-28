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

## Correction: the greedy output is not reproducible, so the agreement numbers above are single draws

The token-parity table above was read as "level 3 changed the tokens". That
reading was wrong, and the repeat-request probe shows why.

Same server, same prompt, `temperature=0`, `ignore_eos`, 8 identical
non-streaming requests, counting distinct token streams:

| config | 10-token prompt, 64 out | C-0000 (7881 in) | C-0003 (6183 in) |
|--------|------------------------|------------------|------------------|
| level 3 (compile + cudagraph) | **3 streams** (4/3/1) | 1 stream (8/8) | **3 streams** (4/3/1) |
| level 0 (eager) | **3 streams** (3/3/2) | 1 stream (8/8) | **2 streams** (5/3) |
| level 0, `ATOM_USE_TRITON_MOE=0` (aiter MoE) | **3 streams** (3/3/2) | 1 stream (8/8) | **2 streams** (6/2) |

So the nondeterminism is not caused by compilation and not caused by the choice
of MoE kernel. It survives on a 10-token prompt, so it is not a long-context or
sliding-window effect either.

Consequence for the correctness gate: five (later ten) identical repetitions of
the same 5-prompt comparison against the same fixed HF reference file give

```
0.334 0.338 0.391 0.547 0.594 | 0.553 0.588 0.503 0.250 0.594
```

i.e. token-position agreement between **0.250 and 0.594, mean 0.469**, on a warm
server. The 59.1% figure quoted earlier was one draw from that distribution, and
the "level 3 changes tokens" delta is smaller than the engine's own run-to-run
spread. Until the nondeterminism is removed, no token-level gate and no
token-level ablation on this port can be measured.

Ruled out so far: compile level (0 and 3 both unstable), MoE kernel (Triton and
aiter both unstable), context length (unstable at 10 tokens in), sliding window
(short prompt has no window effect). Still suspect: the paged-attention /
KV-block-layout path, where the reduction order can depend on the block
allocation chosen per request. That is the next thing to pin down.

### Where exactly does the nondeterminism start

Same 10-token prompt, 10 identical requests, output length bisected:

```
max_tokens =  1  2  4  8 12 16 20 24 32 48   -> 1 distinct stream / 10 runs each
max_tokens = 64                              -> 3 distinct streams (4/4/2)
```

Collecting 10 runs at `max_tokens=64` and locating the first index where the
runs disagree:

```
common prefix over all 10 runs: 55 generated tokens
prompt is 10 tokens, so the first unstable prediction is at context length 65
```

Every divergent run agrees with the others for the first 55 generated tokens and
diverges at exactly index 55 - i.e. as soon as the context passes 64 tokens.
That is not a tie-breaking coin flip spread over the sequence; it is one
specific step that becomes unstable once the context crosses 64.

What 64 is *not*: `context_partition_size` in the paged-ASM path is 128 or 256,
so it is not the attention context partitioning. The KV block size in this run is
16, so 64 tokens is 4 blocks - the block-allocation/block-table path remains the
best hypothesis, but it is a hypothesis, not a verified cause. The next concrete
test is to change `kv_cache_block_size` and see whether the unstable index moves
with it; if it stays at 55, the threshold belongs to something else entirely.

Also worth noting: the nondeterminism only becomes *visible* when the step lands
near a tie, which is why C-0000 was stable 8/8 while C-0003 was not. So
"C-0000 is stable" is luck, not evidence of a working path.
