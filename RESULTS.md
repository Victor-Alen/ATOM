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

### Block-size sensitivity

Repeating the 10-request, 96-token probe at three KV block sizes (level 0,
eager, everything else identical):

| `--block-size` | distinct streams / 10 runs | first unstable generated index | context length there |
|----------------|---------------------------|-------------------------------|----------------------|
| 16 (default) | 3 | 55 | 65 |
| 32 | 10 | 1 | 11 |
| 64 | 10 | 1 | 11 |

Changing the block size does not just move the 64-token threshold - it makes the
output unstable from the very first decoded token. That means the block size
selects a different attention kernel path rather than only changing the layout,
so this probe does not isolate the block allocator on its own. It does establish
one practically useful fact: the default `block_size=16` is the *least* unstable
configuration of the three, and any tuning that touches the KV block size has to
re-run the determinism probe before its speed numbers mean anything.

Open question, stated as a question and not a conclusion: with `block_size=16`,
why does instability begin exactly when the context passes 64 tokens? Candidates
not yet excluded: the paged-ASM kernel's partition count as a function of
context length, and the block-table walk when the row count crosses a kernel
tile boundary. Answering it needs a kernel-level read, not more black-box runs.

### Prompt-length sweep - the "64 token threshold" was a coincidence

Same server, same 10 identical requests per length, 96 tokens out, prompts are
prefixes of one long scenario-C prompt so the content stays comparable:

| target prompt tokens | actual prompt tokens | distinct streams / 10 | first unstable generated index | context length there |
|----------------------|----------------------|-----------------------|-------------------------------|----------------------|
| 20 | 21 | 3 | 4 | 25 |
| 60 | 61 | 1 | - | - |
| 130 | 130 | 1 | - | - |
| 300 | 294 | 1 | - | - |
| 1000 | 997 | 4 | 1 | 998 |
| 3000 | 2989 | 5 | 1 | 2990 |

This kills the "fixed 64-token threshold" reading of the earlier bisect:

- The 61-token prompt runs 96 decode steps, reaching context length 157, and is
  stable in all 10 runs - it crosses the alleged 65 boundary without a single
  flip. So there is no universal 65-token edge.
- At ~1k and ~3k prompt tokens the *first or second* decoded token already
  differs between runs, while at 130 and 294 tokens a full 960 decode steps are
  bit-identical. That is not a boundary effect, it is a steep growth of
  per-step numerical divergence with context length.
- The earlier "common prefix is exactly 55 for a 10-token prompt" was one
  prompt's first near-tie, not a structural threshold. The 64 in that run
  coincided with a real near-tie, which is exactly the kind of coincidence that
  looks like a cause if you stop after one prompt.

Revised picture: per-step numerical divergence exists at all lengths but only
changes the output when a step also lands near a tie. At >= ~1k context the
divergence is large enough to flip the very first token, so ties are no longer
required. That points at the attention reduction over KV whose split/partition
count grows with context, not at KV block boundaries or a length constant.

Next probe, if this is picked up again: read how `paged_attention_asm` /
`paged_attention_triton` choose their split count as a function of context
length, and check whether any of those splits accumulate with atomics.

### The port is exact at short context (and the 128-token-coverage theory is wrong)

`paged_attention_triton`'s docstring says
`max_context_partition_num = ceil(max_context_length / context_partition_size)`,
and ATOM passes `max_context_partition_num = 1` with `context_partition_size = 128`
for sliding layers. Read literally that caps sliding-layer coverage at 128 keys
instead of the 1024-token window, i.e. "the kernel never reads the earlier
blocks". That was worth testing, so:

284-token prompt (prefix of C-0003), HF eager reference, 64 greedy tokens,
ATOM compared 5 times:

```
rep1..rep5: agreement = 1.0000, 64/64, exact sequence 1/1 every time
```

So the theory is **wrong**: at a context of 284 tokens - well past 128 - ATOM
reproduces HF bit for bit. The kernel does loop over the whole context; the
partition count is a parallelism tiling parameter, not a coverage limit.

Two conclusions, both load-bearing:

1. The Mellum integration itself is correct: checkpoint weight mapping, the
   3:1 sliding/full layer pattern, per-attention-type RoPE and YaRN, QK-norm,
   and top-8 MoE routing together reproduce HF exactly in the regime where the
   engine is deterministic. The earlier ~50% numbers were never measuring the
   port's correctness.
2. Everything that is still wrong is specifically a **long-context numerics**
   problem, not a wiring problem. That is a much narrower target.

The ranges line up cleanly:

| prompt tokens | engine determinism | agreement with HF |
|---------------|--------------------|-------------------|
| 284 | bit-identical over 5 runs | 100% (64/64) |
| 997 / 2989 | diverges at the first token | not measurable |
| 6183 / 7881 / 11889 / 12580 | diverges within 1-3 tokens | 25-59% across repeats |

Next target is therefore the multi-partition softmax combine of the long-context
decode path: `exp_sums`, `max_logits` and `temporary_output` are allocated with
`torch.empty` (uninitialised) in `paged_attention_triton`, and the combine over
up to 8 partitions is where reduction order could vary. That is the concrete
place to look, and it is now the only place that matters for the gate.
