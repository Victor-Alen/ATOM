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

### Actual context-partition counts on this GPU (MI308X, 80 SMs, occupancy 2)

`get_recommended_splits` = `min(cdiv(80 * 2, num_seqs * 4 * 1), 8)`:

| attention layer | max_context_partition_num | context_partition_size | covered per grid pass |
|-----------------|---------------------------|------------------------|-----------------------|
| full attention, 1 request | 8 | 256 | 2048 tokens |
| full attention, 32 concurrent (scenario A) | **2** | 256 | 512 tokens |
| sliding attention, any concurrency | **1** | **128** | 128 tokens, window start = ctx-1024 |

Three things worth flagging, all read out of the code rather than inferred:

1. The partition count depends on **concurrency**, not just context: 8 at one
   request, 2 at 32. So the reduction grouping - and therefore the numerics -
   differ between scenario B/C and scenario A. Correctness established at
   concurrency 1 does not transfer to concurrency 32 by construction.
2. `pa_decode_gluon`'s kernel body asserts
   `CONTEXT_PARTITION_SIZE == 256, "Only support CONTEXT_PARTITION_SIZE == 256"`,
   yet ATOM passes **128** for sliding layers. Either that assert belongs to a
   variant this path does not hit, or the 128 configuration is unsupported and
   silently diverted. Needs a runtime check of which kernel actually launches.
3. The sliding path computes `sequence_partition_idx = (ctx - 1024) // 128 + pid`
   and then a hardcoded `page_offset` from `sequence_partition_idx % 4`. With
   one partition the `pid` term is always 0, so only the page containing the
   window start is visited; the "4" implies a 512-token page for size 128.
   Whether the dynamic loop then covers the remaining window is not established
   by reading alone.

Consistency check against the measurements: 284 tokens matched HF exactly
(the combine spans at most 2 partitions there), while 997 tokens is the first
length where four full partitions participate in the softmax combine - and 997
is exactly where the run-to-run divergence appears at the first token. That is
correlation, not proof, but it is the sharpest hypothesis on the table.

### Attempting to isolate the sliding-window fix (window=16) - inconclusive

Goal: with the real window of 1024, short prompts never activate the sliding
mask, and contexts above 1024 are not reproducible, so the off-by-one fix in
`5b72cc44` could not be validated either way. Plan: copy the config with
`sliding_window = 16`, use 101-169 token prompts (window active, context still
small), generate an HF reference under the same config, then compare ATOM with
the fix against ATOM with the fix reverted.

Results, 3 prompts x 3 runs each, 64 tokens, same HF reference:

| build | agreement per run | first generated token correct | mean common prefix |
|-------|-------------------|-------------------------------|--------------------|
| fix present | 0.552 / 0.365 / 0.115 | 3/9 | 14.7 / 64 |
| fix reverted | 0.026 / 0.115 / 0.318 | 1/9 | 3.8 / 64 |

Verdict: **inconclusive, and the design did not work as intended.**

- Both builds disagree with HF on the first generated token most of the time
  (6/9 and 8/9 wrong), while the same port at window=1024 and 284 tokens matched
  HF perfectly 5/5. Dropping the window to 16 therefore introduced a *larger*
  discrepancy that swamps the one I wanted to isolate. `SLIDING_WINDOW` is a
  compile-time constexpr inside the decode kernel, so window=16 selects a
  different kernel variant - I changed the experiment's regime, not just the
  boundary condition.
- Same build, same prompts, different runs still disagree (0.115 to 0.552), so
  the reproducibility problem is not fixed by short prompts either.
- The direction does favour the fix (first token 3/9 vs 1/9, common prefix 14.7
  vs 3.8), but 9 samples with this spread is not evidence. Recorded as a hint,
  not a result.
- One new lead: prompt SW16-C (169 tokens) produced `atom0=1622` while the HF
  reference was `1253` in **all six** runs across both builds. That is a
  repeatable, build-independent disagreement, unlike everything else measured so
  far. Worth chasing on its own.

What would actually settle the question: make the engine deterministic first
(then window=1024 with 4k contexts becomes measurable directly), or compare
logits instead of tokens at a single position, where a one-key window
difference shows up as a small numeric delta rather than a coin flip.

### Determinism hunt, round 1: four candidates, all rejected

Judge: 8 identical greedy requests on 3 prompts, counting distinct token
streams. Baseline today: short64 = 3, C-0000 = 1, C-0003 = 2-3.

| variant | what it changes | short64 | C-0003 | verdict |
|---------|-----------------|---------|--------|---------|
| V1 | sliding layers use partition size 256 instead of 128 | 3 | 6 | no effect, worse |
| V2 | force `max_context_partition_num = 1` on every layer | 2 | 4 | no effect |
| V3 | partition buffers zero/`-inf` initialised instead of `torch.empty` | 3 | 2 | no effect |
| V4 | `use_flash_layout = True` (unified attention instead of gluon paged) | - | - | not viable |

What this rules out, which is worth more than the failed fixes:

- **The multi-partition softmax combine is not the cause.** V2 removes partition
  parallelism entirely and the output still varies run to run. My earlier
  hypothesis about `exp_sums` / `max_logits` combining 4-8 partitions was wrong.
- **Uninitialised buffers are not the cause.** V3 gives the buffers defined
  contents and nothing changes.
- **The 128 vs 256 partition size is not the cause.** V1 keeps the instability.
- **`use_flash_layout` is not a switch.** It requires the KV cache to be
  allocated in flash layout by the backend; flipping the flag alone dies in
  `fused_qk_rope_reshape_and_cache` with `ValueError: too many values to unpack
  (expected 4)` because the cache is still 5D. Using that path would mean
  changing the KV cache builder too.

Still standing as suspects, now a short list: a reduction inside the gluon paged
kernel over KV blocks, the MoE gating top-k tie-breaking, or aiter's GEMM
solution selection (the log shows "not found tuned config ... using torch
solution:0", which means the chosen solution can depend on a runtime lookup).

Process note: the automated variant loop left the source tree patched and a
server that answered `/health` but returned nothing for a real completion. Both
were caught only by running an actual generation request, and both were fixed.
`/health` is not evidence that a server can serve.

### Determinism hunt, round 2: the one positive result did not reproduce

Round 1 ruled out the partition combine. Round 2 tested the MoE gating top-k,
replacing the aiter `topk_softmax` kernel with a torch reference
(`softmax` -> `topk` -> renormalise).

| run | patches | short64 | C-0003 |
|-----|---------|---------|--------|
| G1 | deterministic gating | **1** distinct / 8 | 2 |
| G2 | deterministic gating + aiter autotune env off | 2 | 3 |
| H1 | deterministic gating + single context partition | 3 | 3 |
| H2 | deterministic gating (identical to G1) | **3** | 3 |

G1 and H2 apply the *same* patch and disagree: 1 distinct stream versus 3. So the
apparent "gating top-k fixed it" result was a single draw, not a fix. This is the
second time in this investigation that one lucky run looked like a cause; the
first was the 64-token threshold.

Confidently ruled out so far (each left the variation in place):

- compile level (0 and 3 both unstable, and level 3 is internally consistent)
- MoE kernel choice (Triton and aiter both unstable)
- context partition count and partition size (V1, V2)
- uninitialised partition buffers (V3)
- MoE gating top-k (G1/H1/H2)
- aiter autotune env toggles (G2)
- a fixed context-length boundary (length sweep: 61 tokens stable through 960
  decode steps, 997 tokens divergent at the first token)
- context length alone as the driver (10-token prompt also flips, at step 55)

**Methodology conclusion, which is the real result of this round:** with 8
identical requests, the distinct-stream count is too coarse an estimator. A step
whose flip probability is low will show 1 distinct stream most of the time and
2-3 occasionally, so any single configuration can look either fixed or broken.
The comparison needs at least ~40 requests per configuration, and the metric
should be the fraction of runs agreeing with the modal output rather than the raw
distinct count. Until then, every "this fixed it" observation in this file
should be treated as provisional - including any that sound encouraging.

### Nondeterminism baseline at n=40 (the measurement protocol that can judge a fix)

Canonical Step 0 server (level 0, eager, bf16 KV, default block size, prefix
cache off). `temperature=0`, `ignore_eos`, 64 tokens out, 40 identical requests
per prompt. `modal_rate` = share of runs producing the most frequent output;
its binomial standard error at n=40 is ~0.08 near p=0.5.

| prompt | input tokens | distinct outputs / 40 | modal_rate | counts (desc) | common prefix |
|--------|--------------|----------------------|------------|---------------|---------------|
| short64 | 10 | 3 | **0.600** +- 0.077 | 24, 10, 6 | 55 / 64 |
| C-0000 | 7881 | 1 | **1.000** | 40 | 64 / 64 |
| C-0003 | 6183 | **14** | **0.225** +- 0.066 | 9, 7, 6, 3, 3, 3, 2, 1x7 | 2 / 64 |

Three things this establishes:

1. **There is a usable control.** C-0000 is 40/40 identical at 7881 input
   tokens. So long context alone does not cause instability - content does. A
   candidate fix must leave C-0000 at 1.000, which makes "fixed it by breaking
   everything" detectable.
2. **There is a usable sensitive case.** C-0003 gives 14 different outputs in 40
   runs with a common prefix of only 2 tokens. Its modal rate 0.225 has SE 0.066,
   so a real improvement has to clear roughly 0.36.
3. **The earlier n=8 protocol was indeed too weak.** short64 with 3 outcomes and
   a 0.600 modal rate means an 8-run sample lands on 1 distinct output about
   `0.6^8 = 1.7%`... in fact higher once near-ties are counted, but the point
   stands: the "1 distinct stream" readings that looked like fixes were tail
   draws, exactly as the G1/H2 contradiction showed.

**Verdict rule for any future determinism change**, adopted from here on:

- run this probe with n>=40 on all three prompts before and after,
- accept the change only if C-0003 modal_rate rises above baseline + 2 SE
  (0.225 -> above ~0.36, ideally toward 1.0),
- and C-0000 stays at 1.000,
- and short64 does not regress.

Everything in this file measured at n=8 should be read as anecdote, not result.

### n=40 rounds for the three surviving suspects: all three rejected

Baseline: short64 modal 0.600 | C-0000 1.000 | C-0003 0.225 (+-0.066).
Acceptance required C-0003 above ~0.36 with C-0000 held at 1.000.

| round | patch | short64 | C-0000 | C-0003 |
|-------|-------|---------|--------|--------|
| baseline | none | 0.600 | 1.000 | 0.225 |
| R1 | deterministic torch gating top-k | 0.500 | 1.000 | 0.275 |
| R2 | single context partition on every layer | 0.475 | 1.000 | 0.250 |
| R3 | aiter fused MoE instead of Triton MoE | 0.500 | 1.000 | 0.250 |

C-0003 moves by at most 0.05 against a standard error of 0.066-0.071, i.e. all
three are within one standard error of the baseline. Nothing here is a fix, and
the n=8 readings that suggested otherwise were tail draws, as suspected.

**What survives, and it is now a short and specific list.**

Two observations are firm because they reproduce across every configuration:

1. The divergence point for the 10-token prompt is **always token index 55**
   (context length 65). Baseline and all three rounds report `common_prefix=55`
   at n=40. Whatever goes wrong is triggered by crossing a 64-token boundary -
   the location is deterministic, only the outcome varies.
2. C-0000 is 40/40 identical at 7881 input tokens in every configuration. Long
   context does not by itself destabilise; content does.

Combined with everything already excluded - compile level, MoE kernel (two
implementations), gating top-k, partition count, partition size, buffer
initialisation, aiter autotune toggles, block size (which makes it worse), and
the flash-layout path (not usable without reallocating the KV cache) - the
remaining suspect is a reduction **inside the compiled gluon paged-attention
kernel**, which I cannot patch from Python. Confirming it needs a kernel-level
read of how that kernel walks its KV blocks, not more black-box runs.

### Where this leaves the correctness gate

The task defines the gate as per-token equality against HF over 200 prompts,
single greedy run each. On this engine that measurement is not stable: the same
request returns 10-14 different outputs per 40 runs on C-0003. So the honest
position for the report is:

- the port reproduces HF **bit-exactly** wherever the engine is deterministic
  (284-token prompt, 64/64 tokens, 5/5 runs, agreement 1.0000);
- the engine itself is nondeterministic at temperature 0 on long prompts, which
  is an ATOM/AITER property, not something the Mellum port introduced;
- therefore the gate must be reported with a repetition-based protocol (modal
  agreement over N runs) or after the kernel-level nondeterminism is fixed, and
  a single-run 90% number would be measuring the engine's coin flips.

## 2026-10-29: read the active decode kernels; Step 0 still open

This section supersedes the prior suggestion that run-to-run output variation is caused by an unordered reduction over KV blocks inside the Gluon kernel. That was a hypothesis, not a confirmed cause.

**Actual dispatch for Mellum, block size 16, BF16 KV, level 0:** sliding layers (21/28) have `sliding_window=1024` and call `paged_attention_triton` -> `torch.ops.aiter.pa_decode_gluon(..., ps=True)`. The AITER wrapper selects `paged_attention_decode_sliding_window` for this configuration. Full layers (7/28) have `sliding_window=-1`, 32 Q heads, 4 KV heads, head size 128, so `use_triton_attn=False` and dispatch to `paged_attention_asm` -> `aiter.pa_fwd_asm`. Earlier discussion treated all layers as Gluon; that was wrong. Sources: `atom/models/mellum.py`, `atom/model_ops/attention_mha.py`, `/app/aiter-test/aiter/ops/triton/gluon/pa_decode_gluon.py`, `/app/aiter-test/csrc/py_itfs_cu/asm_pa.cu`.

**Gluon block walk:** `paged_attention_decode_sliding_window` computes a sequence partition range, looks up physical KV block numbers through `block_tables`, then loops over the partitions in increasing order. Each loop iteration computes QK scores, applies the sequence/window mask, updates `max_logits` and `exp_sums` with online softmax, rescales the FP32 accumulator, and adds the PV result. The sliding path uses one program per (sequence, KV head) and writes the output directly. No atomic accumulation over KV blocks appears in this path. The full path, if explicitly forced to Gluon, has a separate fixed-order split-reduction kernel, but it is not the default Mellum full-attention path here.

**Fixed-input direct kernel checks:** identical Q/K/V, context length and block table were passed directly to the same AITER entry points 40 times per case. Gluon: context 65 at sliding 1x128; 997 and 6183 at sliding 1x128, full 1x256 and full 8x256: all seven cases produced one bit-identical output (max difference 0). ASM `pa_fwd_asm`: contexts 65, 997, 6183, 40 repeats each, also one bit-identical output (max difference 0). These checks do not prove either kernel numerically matches HF, but they reject intrinsic run-to-run randomness for fixed inputs in the tested configurations.

**A separate sampler defect exists, but is not the complete explanation.** At `temperature=0`, `prepare_sample` sets `all_greedy=True`, yet the original `Sampler.forward` without top-k/top-p called `_temperature_sample`, which draws a fresh exponential random row. The current uncommitted patch routes `all_greedy` to `argmax`; it should be retained only after its own correctness check. The run tagged `allgreedy` still reported short64 3 outputs/40 (modal 0.725) and C-0003 13 outputs/40 (modal 0.375), so the patch has not established determinism. The currently traced server also saw `all_greedy=true`, `filtering=false`, and the short prompt's divergence step had rounded top-2 gap 0 while `argmax` itself differed (1855 versus 1988) across requests. Therefore logits entering the sampler can differ before sampling; a sampling fix alone cannot complete Step 0.

**Next discriminator:** instrument the attention/model call boundary for repeated identical requests, logging exact hashes of each layer's Q, the logical valid KV entries and block table, and the output, starting at the short prompt's context 65 and at C-0003's first decoded token. Compare the first layer where inputs or outputs diverge. If attention inputs are identical but outputs differ, inspect that layer's active kernel and memory safety; if inputs differ, walk backward through QK norm, KV write, MoE/GEMM and prefill. Avoid more block-size or split-count sweeps before this boundary is located. Continue to regard Step 0 as incomplete until the 200-prompt HF token gate is measured reproducibly.

**Corrections to two earlier source readings:** the `CONTEXT_PARTITION_SIZE == 256` assertion and `sequence_partition_idx % 4` page-offset logic are in the large-block dot-kernel variant, not the selected `paged_attention_decode_sliding_window` variant at KV block size 16. They do not establish a 128-vs-256 incompatibility for the Step 0 sliding layers.

**Trace attempt status:** a separate detached debug checkout at `/workspace/ATOM-kvtrace` has opt-in Q/logical-KV/block-table/output hashing at context 65 and a local greedy argmax guard. Its service requested visible device 0 on port 8001, loaded and reached ready, then received a clean shutdown at 11:59:15 UTC while another process was restarting the 8000 service. No per-layer trace was captured. The main `/workspace/ATOM` attention file was restored exactly; only this RESULTS edit was added by this investigation. Do not interpret 8001 readiness as a completed trace or a Step 0 pass.

**Physical block-table permutation check:** with the same logical Q/K/V, all physical KV blocks were permuted and the block table updated to preserve the logical sequence. Across 20 permutations each at context 65, 997 and 6183, both the active sliding Gluon kernel and full-attention ASM kernel produced exactly one bit-identical output (max difference 0). Thus physical block allocation order alone does not explain the observed repeat-request variation in these cases. This still does not exclude an upstream cache write race or different logical KV contents.


## 2026-10-29 (round 2): the Gluon KV-block reduction is exonerated; the divergence is already in the prefill logits

This section is additive to the one above: it settles the "unordered reduction over KV blocks" hypothesis with direct kernel calls instead of more black-box sweeps, and it moves the open question to the model forward. Artifacts: `scripts/kernel_pa_probe{,2,3}.py`, `scripts/sampler_probe.py`, `scripts/patch_sampler.py`, log `/workspace/logs/sampler_trace.txt`, benchmark `/workspace/bench/det40_allgreedy.json`.

**Dispatch re-verified (this is per layer, not global).** `atom/model_ops/attention_mha.py:135-140` sets `use_triton_attn = (self.sliding_window != -1 or self.head_dim != 128 or self.num_heads == self.num_kv_heads)`. For Mellum: 21/28 sliding layers have `sliding_window=1024` -> `use_triton_attn=True` -> decode via `paged_attention_triton` -> `torch.ops.aiter.pa_decode_gluon(..., ps=True)` -> the PS branch of `_paged_attention_decode_v2_with_dot_kernel_reshape_wrapper` -> `paged_attention_decode_sliding_window`. The 7/28 full layers have `sliding_window=-1`, `head_dim=128` and 32/4 heads -> `use_triton_attn=False` -> `paged_attention_asm` -> `aiter.pa_fwd_asm` (block size 16 != 1024, so not the persistent variant). The ASM path owns the full-attention layers; the Gluon kernel owns the sliding layers.

**What the Gluon block walk actually does (read from source, `paged_attention_decode_sliding_window`, aiter at /app/aiter-test).** Grid is (sequence, kv_head, context_split). For the sliding layers ATOM passes `max_context_partition_num=1` and `context_partition_size=128`, so `ONE_SHOT=True`: one program per (sequence, KV head) writes the output directly and no reduce kernel runs. With `KV_BLOCK_SIZE=16 <= CONTEXT_PARTITION_SIZE=128`, `KV_COMPUTE_BLOCK_SIZE=KV_BLOCK_SIZE=16`, `MAX_NUM_KV_BLOCKS_PER_COMPUTE=cdiv(128,16)=8`, and `max_num_kv_blocks=cdiv(context_length, KV_COMPUTE_BLOCK_SIZE)`; the physical block ids come from `block_tables` under the mask `block_indices < (max_num_kv_blocks - kv_block_start_idx)`. The following `for sequence_partition_idx in range(start, end, 1)` walks partitions in strictly increasing order; each iteration does QK MFMA, applies the mask (column < context_length, plus the sliding-window lower bound), fills masked lanes with -3.4e38, takes `max`, rescales the FP32 accumulator and `exp_sums` by `exp2((old_max - new_max) * log2 e)`, then accumulates PV. This is a sequential online softmax: no atomic accumulation, no cross-CTA combine and no data-dependent visit order on this path. The only split reduction in the file belongs to the non-ONE_SHOT geometry (8 splits x 256-token partitions -> `paged_attention_decode_v2_reduce_kernel`), which is what ATOM would use for a full layer only if that layer had been routed to Gluon.

**Fixed-input direct kernel calls (GPU 0, ATOM geometry q=32/kv=4/head=128/block=16), through the same python entry point that registers the op.** 48 cases = 16 context lengths (16, 32, 48, 63, 64, 65, 80, 96, 127, 128, 129, 192, 255, 256, 257, 300) x 3 configurations (sliding 1x128, full 1x256, full 8x256), with 8 identical launches per case:
- determinism: every case returned exactly one bit-identical result (distinct=1, max difference 0), including the 8-split reduce path;
- correctness: against an independent FP32 reference the max abs difference was 2.0e-3 to 8.0e-3, i.e. bf16 resolution, at every context length including 64/65/128/129/256/257 - the 64-token boundary is not mishandled;
- out-of-context reads: rewriting every KV entry at token index >= context_length (the tail of the last partial block plus all blocks outside the valid range) with fresh random values changed the output by exactly 0.0000 at contexts 17, 33, 48, 64, 65, 66, 80, 96, 100, 129.
A leak at context 65 that appeared in my first measurement was an artifact of my own scatter: the paged layouts are key `[num_blocks, num_kv_heads, head_size/8, block, 8]` and value `[num_blocks, num_kv_heads, block/8, head_size, 8]` (matching aiter's own `create_kv_cache` and `shuffle_value_cache_layout`), and my initial scatter used token-major order. With the correct scatter both the leak and the O(1) reference mismatch disappeared. Do not cite the first probe run.

**Where the run-to-run variation really starts.** An env-gated trace in `Sampler.forward` (`ATOM_SAMPLER_TRACE`) records rows, `all_greedy`, whether filtering is on, temperatures, argmax and the top1-top2 logit gap for every sampling call. Four identical greedy requests to one server process (the 10-token python-function prompt, max_tokens 64): the prefill step itself already differs - top1-top2 gap 6.53125 / 6.515625 / 6.578125 / 6.625 with argmax 255 in all four. Argmax then agrees for 55 tokens and the runs split from there (3 distinct streams, common prefix 55). So the model forward is nondeterministic from the first sampled token on identical input, and `common_prefix=55` is only the point where the surviving margin becomes small enough. This is independent of, and fatal to, any explanation that blames the KV-block reduction.

**A real and separate sampler defect (patch kept uncommitted).** At temperature 0 with no top-k/top-p, `Sampler.forward` routed to `_temperature_sample`, a Gumbel/exponential race that redraws `torch.empty(...).exponential_(1)` on every call; `all_greedy` was honoured only inside the top-k/top-p branch. Unit check on fixed logits (`scripts/sampler_probe.py`): an exact two-way tie at the maximum gave a 13/7 split over 20 calls, a three-way tie gave 3 outcomes, a 1-ulp gap gave one outcome, and with the argmax patch the result is a single outcome at any tie arity. Because bf16 logits resolve about 1e-2 near |x|=10, exact ties at the top are ordinary, so before the patch "temperature 0" was never a true argmax. With the patch, the n=40 protocol at level 0 still gives short64 3 distinct (modal 0.725), C-0000 1.000, C-0003 13 distinct (modal 0.375): the patch fixes the sampler, not Step 0.

**Next discriminator (same protocol as this round).** Apply the fixed-input bitwise test to `aiter.fused_moe` (ATOM dispatches to it: 64 experts, top-8, moe_intermediate 896, `norm_topk_prob=True`) - 50 identical calls, compare bytes. If it is nondeterministic, the prefill divergence is explained and the fix is a deterministic MoE accumulation. If it is deterministic, hook the model per layer on the prefill step and report the first layer whose output hash differs between two identical requests. Do not spend further rounds on block size, split count or partition size until that layer is named.

**State.** Branch `mellum-step0`. `atom/model_ops/sampler.py` carries the argmax patch plus the env-gated trace block (+26 lines, uncommitted, trace off unless `ATOM_SAMPLER_TRACE` is set); the probe scripts are committed together with this section. The port-8000 service was restarted at level 0 with a clean environment and verified with a real completion request. Step 0 remains incomplete, and the repeated-request nondeterminism is still an ATOM forward issue rather than a Mellum port issue as far as this round can tell.
