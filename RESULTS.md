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


### 2026-10-29 Step 0 update: source-level localization to BF16 MoE atomic reduction

The fixed-input KV experiments and source walk rule out the earlier Gluon KV-block reduction hypothesis for the observed first divergence. In 9 identical 64-token debug requests, context 65 layer 0 attention input Q/K/V, logical cache K/V, and attention output were bit-identical despite differing physical KV block IDs. Layer 1 inputs differed. A forward-hook trace of 5 requests showed layer 0 attention, RMSNorms and router linear output identical, but layer 0 FusedMoE output had 5 distinct hashes (maximum absolute difference 0.00390625). Further opt-in tracing of 5 requests showed identical layer 0 MoE input, router logits, selected expert IDs [5,1,31,46,7,49,55,23], and top-k weights, yet 5 distinct MoE outputs. This is a pre-sampler, post-routing variation inside the active AITER non-modular BF16 fused MoE path.

Source chain: `atom/model_ops/moe.py` UnquantizedFusedMoEMethod.apply calls `aiter.fused_moe`; `/app/aiter-test/aiter/fused_moe.py` chooses the two-stage path for this decode shape. Stage 2 dispatches `ck_moe_stage2_fwd` (`csrc/py_itfs_ck/moe_ck_2stages_kernel.cu:490-506`). In CK `3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp:280-281`, non-input GEMM selects `InMemoryDataOperationEnum::AtomicAdd`. In the stage-2 launcher, `KBatch=1` (`csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm_common.cuh:186`), and the CK invoker only clears output when KBatch > 1; Opus sorting clears the MoE output buffer instead (`csrc/include/moe_sorting_opus.h:1100-1103`). Thus selected experts atomically accumulate BF16 output, and their arrival order can alter rounding. This is a source-supported mechanism consistent with the observed output variation. It is not yet a confirmed full fix.

An isolated debug worktree `/workspace/ATOM-kvtrace` now contains an opt-in `ATOM_MOE_DETERMINISTIC=1` reference path that inverts the AITER weight shuffle, computes each selected expert with PyTorch, and sums weighted outputs in a fixed order in FP32. This is for causal validation, not a performance-ready solution; it may differ numerically from CK/HF. Main worktree model code remains untouched by this experiment. Step 0's 200-prompt per-token HF correctness gate remains outstanding.


## 2026-10-29 (round 3): layer-by-layer bisect names the aiter 2-stage CK MoE as the nondeterminism source

The prefill forwards of two identical greedy requests differ, so the divergence was bisected inside the model. Instrumentation: `atom/models/layer_trace.py` (env-gated by `ATOM_LAYER_TRACE_DIR`, installed from `MellumModel.__init__`), 168 forward hooks = 28 layers x (input_layernorm, self_attn, post_attention_layernorm, mlp, mlp.gate, layer). Each forward appends one JSON line per hook with a sha1 digest of the bf16 bytes plus absmax/mean/last-row fingerprints. Probe driver: `scripts/wire_layer_trace.py`; MoE probe: `scripts/patch_moe_repeat.py`.

**First divergent operator is inside layer 0's MoE block.** Two identical requests (10-token prompt, max_tokens 4, level 0, enforce-eager), prefill step compared item by item in execution order: items 0-7 are bit-identical - including `L0.input_layernorm.out`, `L0.self_attn.out`, `L0.post_attention_layernorm.out`, `L0.mlp.gate.out` (router logits, [10,64]) and `L0.mlp.in` ([10,2304]) - and the first differing item is `L0.mlp.out` (mean -0.00141365 vs -0.00141594, same absmax 6.5). Every later layer differs because it inherits that difference. So attention, qk-norm, RMSNorm and the router are clean; the expert block is not.

**Inside the MoE, routing is stable and the expert computation is not.** An env-gated probe (`ATOM_MOE_REPEAT`) runs inside `FusedMoE.forward_impl` after the real call: it recomputes `FusedMoE.select_experts` twice and calls the resolved expert path 5 more times with the *same* hidden states, the *same* router logits and the *same* `topk_ids`/`topk_weights`. Result over 84 probed calls (28 layers x prefill-10-rows and 3 single-row decode steps): `route_distinct = 1` in 84/84 (gating and top-k are deterministic) and `fixed_out_distinct = 5` in 84/84 - five repeats of the expert GEMM + combine returned five different byte digests every single time, for both the 10-row and the 1-row shapes. This is the nondeterminism, reproduced without the engine and without the sampler.

**Which kernel that is.** The path is `atom/model_ops/moe.py` `FusedMoE.forward` -> `torch.ops.aiter.moe_forward` -> `forward_impl` -> `UnquantizedFusedMoEMethod.apply` -> `aiter.fused_moe.fused_moe` (probe records `path=aiter_fused_moe`, i.e. not ATOM's Triton experts). The server log shows the concrete dispatch: `run_1stage = False`, `[fused_moe] using 2stage default for (80, 16384, 2304, 896, 64, 8, Silu, bf16, bf16, bf16, QuantType.No, True, False)`, with modules `module_moe_sorting_opus` and `module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2`. So the nondeterminism lives in aiter's 2-stage CK MoE (expert sorting plus the stage-2 accumulate), not in Mellum's port, not in the gluon/asm attention, and not in the sampler.

**Consequences for the earlier hypothesis chain.** The KV-block-reduction hypothesis is dead (round 2). The sampler defect is real but separate (round 2). The remaining single root cause of "temperature 0 is not reproducible" is this MoE accumulation, whose per-call variation is ~1e-6..1e-5 in the block output and then amplifies through 28 layers until an argmax flips - which is exactly the observed pattern of a deterministic divergence token (55) with random outcome.

**Next step, in order of cheapness.** (1) Test `triton_kernel_fused_experts` (atom/model_ops/fused_moe_triton.py) on the same fixed inputs: 5 repeats, if it is byte-stable, route the MoE through it - a Python-level switch, no compiled kernel changes. (2) If aiter must be kept, test the 1-stage path and `moe_sorting_dispatch_policy` variants for a byte-stable combination, and check aiter's stage-2 epilogue for atomic accumulation into the output. (3) Re-run the n=40 protocol only after one of those lands.

**State.** `atom/models/layer_trace.py` and the env-gated install hook in `atom/models/mellum.py` are committed; `scripts/patch_moe_repeat.py` (applies the env-gated repeat probe to `atom/model_ops/moe.py`) is committed as a re-appliable patch, and the patched `moe.py` itself is left uncommitted. The `sampler.py` argmax patch is still uncommitted. The 8000 service was restarted at level 0 without any probe environment variable after the measurements.

### Correction to the Triton-MoE follow-up (same day)

`ATOM_USE_TRITON_MOE=1` does **not** switch this model to the Triton MoE: the flag is read only in `Mxfp4MoEMethod` (`atom/model_ops/moe.py:753`), while Mellum's bf16 path uses `UnquantizedFusedMoEMethod.apply`, which calls `aiter.fused_moe` unconditionally (`atom/model_ops/moe.py:578-620`). A run with that variable set still dispatched aiter (`[fused_moe] using` appears three times in its log) and still first diverged at `L0.mlp.out` (`39d826db235d` vs `85cdde8ffc88`), which is a consistency check of the aiter finding, not a measurement of the Triton path. Testing `triton_kernel_fused_experts` therefore requires wiring it into `UnquantizedFusedMoEMethod` (routing data via `fused_routing_from_topk_triton`) or calling it directly with constructed routing structures; no conclusion about Triton MoE determinism exists yet.


### 2026-10-29 Step 0 follow-up: prefill MoE atomic accumulation verified by interventions

The active AITER BF16 CK stage-2 kernel performs `AtomicAdd` to the expert output buffer (CK `device_moe_gemm.hpp:280-281`). An isolated debug worktree (`/workspace/ATOM-kvtrace`, port 8001) tested two opt-in interventions; the main service on port 8000 and its model code were not changed for these tests.

1. A fixed-order PyTorch reference path for selected experts, with AITER's weight shuffle inverted, gave five identical 64-token completions and identical attention/KV hashes for a 10-token prompt when used in both prefill and decode. Restricting it to decode alone left three token-55 outcomes; at context 11 the layer-1 prompt KV hashes differed across five requests while layer-0 current Q/K/V and layer-0 MoE output were identical. Thus the first random variation was stored during prefill and propagated through history KV. The reference path is deliberately capped at prefill <=16 tokens and is diagnostic only.
2. A closer AITER path runs each top-k slot as a separate `fused_moe` call with topk=1 and accumulates the resulting BF16 expert contributions in a fixed FP32 sum. This avoids multiple expert atomics targeting the same token. Three repeated short-prompt runs had identical 64-token streams. Two consecutive five-prompt Scenario C passes (ten requests total) produced the same token sequence for every corresponding prompt, including long prefill lengths 6,183-12,580. In the first pass against `hf_scenario_c_first5_eager.jsonl`: C-0000 64/64, C-0001 7/64 (first mismatch 5), C-0002 0/64 (first mismatch 0), C-0003 29/64 (first mismatch 29), C-0004 64/64; aggregate 164/320 = 51.25%. The second pass gave exactly the same counts and ATOM sequences. The 90% per-token gate still fails. These results support BF16 atomic accumulation as the reproducibility root cause but show that deterministic accumulation alone does not establish HF agreement.

Artifacts: `/workspace/correctness/separate_vs_hf_eager_first5{,_r2}/`, `/workspace/logs/pa_trace_detpa_early.jsonl`, `/workspace/logs/pa_trace_detall.jsonl`. The isolated worktree has debug-only `ATOM_MOE_SEPARATE_EXPERTS=1` and `ATOM_MOE_DETERMINISTIC=1` branches in `atom/model_ops/moe.py`; these are not performance-ready production changes. The short `hf_smoke.jsonl` reference has a known packed-input issue and cannot validate this path; the eager Scenario C reference is the relevant comparison. Next: diagnose the remaining HF mismatch on deterministic C-0001/C-0002/C-0003 using first-token/logit or per-layer comparison, then make a performant deterministic MoE reduction and rerun the 200-prompt gate.


### 2026-10-29 Step 0 follow-up: HF backend and execution-protocol sensitivity; ordered BF16 control

The saved 200-prompt HF runs used different attention backends, `sdpa` and `eager`. They agree at 3,469/12,800 = 27.10% of the first 64 generated positions, with 28/200 exact sequences. This measures backend sensitivity, **not** run-to-run nondeterminism of one HF configuration. Two independently saved eager generation files agree on all five overlapping 64-token sequences.

The earlier `/workspace/correctness/margin_probe_first5.json` compares a fresh HF teacher-forced full-prefix forward with `use_cache=False` (`scripts/margin_probe.py:82-93`) to saved HF autoregressive generation with `use_cache=True` (`hf_reference_generate.py:141-148`). Its "self disagreement" labels therefore compare different execution protocols. In particular C-0002's full-prefix top-1 token 233 versus saved generated token 3925 does **not** establish HF repeatability failure. The previous wording of this section incorrectly inferred that failure from these data; this paragraph corrects the record. The specified saved HF stream remains a valid target for the stated gate, subject to documenting its backend and generation protocol.

A diagnostic branch summed topk=1 AITER outputs in ascending expert ID order with BF16 rounding to imitate `reference/hf_mellum/modeling_mellum.py:308-318`; it scored 160/320 = 50.0% against the saved eager first five, versus 164/320 = 51.25% with fixed FP32 sum. Adding a PyTorch softmax/topk route and BF16-rounded weights scored 166/320 = 51.875%. Neither is a sufficient accuracy fix. These variants are debug-only, and `/workspace/correctness/separate_hf_order_actual_first5/` and `/workspace/correctness/separate_hf_routing_order_first5/` record the actual runs. An earlier scratch `separate_hf_order_vs_hf_eager_first5` directory predates the hf_order branch correction and is not an hf_order measurement.

Next, compare HF and ATOM logits on identical token prefixes while explicitly distinguishing HF full-prefix `use_cache=False` from cached autoregressive `use_cache=True`. Inspect top-two gaps and then the first layer where numerical drift appears. The eight-call deterministic MoE path is diagnostic and too slow for the final performance submission.


### 2026-10-29 Step 0 experiment ledger and reproducibility notes

All rows below used Mellum BF16 weights and KV cache, compile level 0, `--enforce-eager`, temperature 0, `ignore_eos=true`, and 64 generated tokens. Port 8001 was an isolated detached checkout (`/workspace/ATOM-kvtrace`); port 8000 was the main checkout. The argmax sampler correction was present in the isolated checkout. No result below is a 200-prompt pass.

| Intervention on port 8001 | Input and repetitions | Observation | Evidence |
|---|---|---|---|
| Original AITER MoE plus tracing | Identical 10-token prompt, repeated prefill/decode; fixed routing and five repeats per call | Layer-0 MoE output varies after equal input and top-k; in 84/84 probed calls, each set of five fixed-input MoE outputs had five byte hashes | `RESULTS.md` round 3; `/workspace/logs/moe_trace_ctx65.jsonl` |
| `ATOM_MOE_DETERMINISTIC=1`, decode only | Five identical short requests | Layer-0 current output equal, but layer-1 prompt KV differed at context 11; three outcomes at token 55. Prefill had already written the variation. | `/workspace/logs/pa_trace_detpa_early.jsonl` |
| Same fixed-order PyTorch reference on prefill <=16 and decode | Five identical short requests | Complete 64-token streams and logical KV hashes equal across all five; diagnosis only, since longer prefill falls back to AITER | `/workspace/logs/pa_trace_detall.jsonl` |
| `ATOM_MOE_SEPARATE_EXPERTS=1`: eight AITER calls with topk=1, FP32 fixed sum | Two consecutive five-prompt Scenario C runs against saved HF eager | Corresponding ATOM streams identical in both runs; saved HF agreement 164/320 (51.25%), below 90% | `/workspace/correctness/separate_vs_hf_eager_first5/summary.json`, `/workspace/correctness/separate_vs_hf_eager_first5_r2/summary.json` |
| `ATOM_MOE_SEPARATE_EXPERTS=hf_order`: sorted BF16 sum; then `ATOM_MOE_HF_ROUTING=1` | Five Scenario C prompts each | 160/320 (50.0%) and 166/320 (51.875%) against saved HF eager; neither fixes agreement | `/workspace/correctness/separate_hf_order_actual_first5/summary.json`, `/workspace/correctness/separate_hf_routing_order_first5/summary.json` |

Reproduce the five-prompt comparison with `python3 /workspace/ATOM/scripts/compare_atom_hf.py --input /workspace/correctness/hf_scenario_c_first5_eager.jsonl --output-dir <new-directory> --url http://127.0.0.1:8001/v1/completions --max-tokens 64 --limit 5` while a chosen isolated variant is running. Keep each variant's output directory distinct. The isolated service was stopped after these runs; the main 8000 service remained running and returned `/health` 200. The main worktree retains unrelated uncommitted `atom/model_ops/moe.py` and `sampler.py` changes; this investigation only appended documentation there. Step 0 is still open.

### 2026-10-29 (round 4): Triton MoE path is deterministic but numerically wrong, so not yet a drop-in fix

Reviewed the other work in this file (round-2/3 sections plus the Step 0 update, intervention, HF-instability and ledger sections added later). Root cause agrees with this measurement chain: aiter's two-stage CK stage-2 GEMM accumulates with `InMemoryDataOperationEnum::AtomicAdd` at `KBatch=1`, so BF16 expert outputs are summed in arrival order. Where the plans differ is the fix: the existing intervention is an eight-call `topk=1` accumulation that is reproducible but ~8x the MoE cost and explicitly marked diagnostic, while the performance-ready deterministic reduction is still listed as outstanding. That is exactly what this round tested, so it proceeded.

**Implementation.** Isolated detached worktree `/workspace/ATOM-tritonmoe` at `c66eb6aa`, service on port 8001, GPU0, level 0, enforce-eager, BF16 KV, `ATOM_TRITON_MOE_EXPERTS=1`. `UnquantizedFusedMoEMethod.apply` gains an env-gated branch (re-appliable via `scripts/patch_triton_moe.py`) that (a) inverts aiter's (16,16) weight shuffle once per layer, (b) transposes to the Triton kernel's `[E, K, N]` (w13) and `[E, I, H]` (w2) layout, (c) builds routing with `routing_from_topk` from the already-computed `select_experts` top-k, and (d) calls `triton_kernel_fused_experts` (matmul_ogs). The sampler argmax guard is present in that worktree.

**Result 1 - determinism, yes.** Three identical 64-token requests returned one distinct stream (`[321, 523, 321, 233, 233, 1804, 42, 941, ...]` in all three). Two independent five-prompt Scenario C passes produced identical counts down to the digit, so the path is reproducible, unlike the aiter atomic path.

**Result 2 - accuracy, no.** Against `/workspace/correctness/hf_scenario_c_first5_eager.jsonl` the identical two passes gave 26/320 = 8.125% token agreement, 0/200 exact sequences, gate failed. The aiter-based deterministic variants in this file score 160-166/320 (50-51.9%). A 6x drop means the Triton path is not merely rounding differently from CK: something in the weight remap, the gate-weight application stage, or matmul_ogs input expectations is wrong. The 8.125% figure is a correctness signal, not an accuracy measurement of the approach.

**Result 3 - cost.** Same warm 64-token request: 3.254 s (19.7 tok/s) on the Triton path versus 1.939 s (33.0 tok/s) on the aiter path, i.e. about 1.7x slower, far cheaper than the eight-call intervention. Performance is therefore acceptable if the numerics can be fixed.

**Next discriminator (bounded).** On one frozen layer-0 MoE input, compare three implementations in one process: aiter `fused_moe`, the Triton `matmul_ogs` path, and the per-expert PyTorch reference with the shuffle inverted. If Triton matches the PyTorch reference and aiter does not, the 8.125% comes from elsewhere in the pipeline and the Triton path is usable; if Triton does not match the PyTorch reference, the remap above is wrong and must be fixed before any gate rerun. Only then is a 200-prompt run worth spending.

**State.** Main worktree: only documentation changed by this round. `/workspace/ATOM-tritonmoe` is a detached debug worktree whose `moe.py`/`sampler.py` edits are uncommitted and reproducible from `scripts/patch_triton_moe.py`. The 8000 service was left running; the 8001 service was stopped after these measurements.

### 2026-10-30 (round 5): three-way MoE comparison - Triton matches the fp32 reference at layer 0/1, and breaks at layer 2

Three implementations were compared on the same frozen layer-0 MoE input inside the live Triton-path service (`/workspace/ATOM-tritonmoe`, port 8001, GPU0, level 0, `ATOM_TRITON_MOE_EXPERTS=1`, probe `scripts/patch_moe_triple.py`, log `/workspace/logs/moe_triple.jsonl`): aiter `fused_moe` (2-stage CK, atomic), ATOM's Triton `matmul_ogs` path, and a per-expert FP32 PyTorch reference on the unshuffled weights with a fixed accumulation order.

| path | prefill 10 rows: repeats | 1-row decode: repeats | max abs diff vs fp32 ref (prefill / decode) |
|---|---|---|---|
| aiter 2-stage CK | 3 runs -> **3 distinct** (spread 0.015625) | 3 runs -> **3 distinct** | 0.03125 / 0.00390625 |
| Triton matmul_ogs | 3 runs -> **1 distinct** | 3 runs -> **1 distinct** | **0.0078125** / **0.000244140625** |
| FP32 per-expert ref | 2 runs -> 1 distinct | 2 runs -> 1 distinct | - |

Reference magnitude at that point is absmax 6.53125, so 0.0078125 is exactly one bf16 step. Two conclusions: the Triton path is reproducible, and it is *closer* to the FP32 reference than aiter is. The weight unshuffle, the `[E,K,N]`/`[E,I,H]` transposes and the `routing_from_topk` reconstruction are therefore self-consistent and accurate at least for layer 0.

**The layer-by-layer gap says the break is not at layer 0.** The same prompt was run once on the Triton path and once on aiter *from the same worktree and the same GPU*, both with the layer trace enabled (`/workspace/logs/layertrace_triton2/`, `/workspace/logs/layertrace_aiter_g0/`). Layer 0's MoE input is bit-identical in both (`L0.mlp.in digest 60d9a6cba3e6`), and the MoE output deltas are:

- L0: mean delta 2.99e-06, absmax delta 0.03125
- L1: mean delta 7.52e-06, absmax delta 0.01562
- **L2: mean delta 3.68e-02, absmax delta 1148.0**
- L3: 6.61e-03 / 87.25, L4: 4.58e-02 / 884.0, ... L26: 2.15e-01 / 588.0, L27: 1.82e+00 / 24642.0

So layers 0 and 1 agree to bf16 noise and **from layer 2 the Triton path diverges catastrophically**. That also explains the earlier full-model result: the 26/320 = 8.125% HF agreement reported in round 4 is not a rounding difference, it is this layer-2-and-beyond bug. The round-4 wording ("numerically wrong") stands, but the fault is localised now: correct at the first two layers, broken afterwards, so a per-layer routing or kernel-configuration condition must differ from layer 2 on rather than a globally wrong weight remap.

**Next step.** Extend the triple probe to layers 0-3 (guard on `.layers.[0-3].`) and check whether the Triton output stops matching the FP32 reference at exactly layer 2, and whether `hist`/`routing_data` or the selected `matmul_ogs` configuration differs there (expert-count histogram, zero-token experts, `M*topk` tile choice). Until that is fixed, the Triton MoE is a reproducible path but not a usable one, and the aiter-side 51% deterministic variants remain the best available reference numbers.

**State.** This round ran entirely in `/workspace/ATOM-tritonmoe` (branch `dsh/moe-determinism`); `/workspace/ATOM` was not modified, and port 8000 was left untouched. The 8001 service was stopped afterwards.
