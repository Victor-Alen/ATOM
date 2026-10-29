# WORKLOG

## D1 - 2026-10-28

- Confirmed the assigned host has two AMD Instinct MI308X GPUs and the frozen Mellum 2.1 weights under `/data/models`. Kept `/app/ATOM` unchanged and used `/workspace/ATOM` at upstream commit `26f23a0b` for the integration work.
- Registered the `mellum` config and `MellumForCausalLM` architecture. Added a Mellum model based on the Qwen3-MoE weight layout, with per-layer full/sliding attention and per-attention-type RoPE/YaRN configuration.
- Started a single-engine ATOM OpenAI server in eager mode at compilation level 0 with bf16 KV cache and a 32,768-token maximum length. Verified `/health` and observed HTTP 200 completion requests with 5, 990, and 7,881 input tokens.
- Reviewed the initial integration and confirmed it already calls ATOM `FusedMoE`; a generic "replace Python MoE loop with fused MoE" item is therefore not a valid later optimization for this Step 0.
- Correctness is not yet certified: the 200-prompt first-64-token HF comparison, separate 16k check, and HumanEval comparison are still pending. No formal scenario A/B/C baseline result has been recorded.

## D2 - 2026-10-29

- Added a native HF Mellum reference path pinned to upstream Transformers commit `ca68c9e07b69cd686c8bbf8869cc40889f5c82ce`, with compatibility shims confined to the local reference modules. The frozen installed packages were not changed.
- Added exact non-streaming ATOM token IDs and a comparison script that records per-prompt agreement, exact 64-token sequence matches, and the 90% token-position gate. Streaming performance traffic is unchanged.
- Diagnosed the initial invalid HF smoke output: frozen Transformers 5.2.0 loaded checkpoint expert shards as unexpected keys while native Mellum's packed 3D expert parameters remained missing. Added explicit gate/up/down checkpoint packing to the reference loader; this does not alter ATOM.
- After the reference fix, the `2+2=` smoke produced identical ATOM and HF token IDs at all 16 positions, with text `4, 2+2=4, 2+2=4,`. This validates the reference path but is not the 200-prompt gate.
- Started greedy 64-token HF generation for all 200 scenario C public prompts. The 200-prompt comparison, separate 16k-input statistics, and HumanEval remain pending; Step 0 is not yet certified.

### D2 (cont.) - 2026-10-29, 01:00-01:30 Beijing time (remote clock is UTC 2026-10-28 17:xx)

- Fixed a real off-by-one in the Mellum prefill attention mask and committed it as `5b72cc44`. aiter/FlashAttention's `window_size=(left, right)` counts keys *outside* the query token, so passing the configured window N as `left` let the varlen prefill path attend to N+1 keys; the decode path in the same file already passed N-1, and the HF reference mask keeps exactly N keys (`kv_idx > q_idx - N`). Only prefill was wrong.
- Wrote `scripts/margin_probe.py` to test something I had been assuming: is the HF reference output a stable target at all? It teacher-forces HF's own 64 reference tokens back through HF eager in a single forward pass and records the top-1/top-2 logit margin at every step. Result over the first five scenario-C prompts (6.2k-12.6k tokens): HF reproduces its own greedy tokens at 313/320 positions (97.8%); 4 of 5 prompts have 1-2 positions where the argmax flips to a different token. Every flipped position has a low top-1 probability (0.14-0.65) and a small logit margin (0.09-0.63). Data: `/workspace/correctness/margin_probe_first5.json`.
- Interpretation kept deliberately narrow: the reference is numerically fragile on these random-token prompts, so per-token agreement will never be a stable 100% target. But ATOM at 59.1% is far below the reference's own 97.8% self-consistency, so this does not excuse the gap - it only bounds how much of it can be blamed on bf16 noise. Remaining candidates, in the order I intend to test: ATOM prefill uses FlashAttention while HF uses a materialised mask, and the MoE reduction order differs; the task's own baseline was measured with "explicit-mask SDPA", which suggests matching that path is what the setter considered reference-equivalent.
- Decision: stop guessing and measure. Started the Step 0 scenario A/B/C baseline against the running server (`--level 0 --enforce-eager --kv_cache_dtype bf16 --no-enable_prefix_caching`, i.e. the task's "most naive port"), using `scripts/bench_scenarios.py` with the exact numbers from the task's own eval commands (A: 512 reqs / concurrency 32 / 256 out; B: 50 reqs / concurrency 1 / 1024 in; C: 200 reqs, open-loop Poisson r=0.30/s). `evalscope` is not installed in the frozen image and the task forbids changing the environment, so I wrote an independent client and will state the difference in the report; the official client is being installed into a separate venv that does not touch `/opt/venv`.

### D3 - 2026-10-29, 01:30-02:00 Beijing time

- Ran the three task scenarios against the Step 0 server (level 0, eager, bf16 KV, MTP off, SWA on, prefix cache off) with a local harness that mirrors the task's own eval commands. 762 requests total, 0 failures. Scenario A: 714.1 tokens/s (published baseline 638.3), TTFT p99 3.47 s vs 5.01 s. Scenario C: 80.9 tokens/s vs 76.2, TPOT p99 42.4 ms vs 41.3 ms. Full table in `RESULTS.md`.
- Deliberately did not celebrate the scenario A gap over the published baseline: my client is not evalscope (which is absent from the frozen image), so the absolute numbers are not bit-comparable. The level-0 vs level-3 rows were measured back to back with the same client and are the ones I trust.
- Started optimization candidate 1: piecewise `torch.compile` + CUDA graph capture (`--level 3`, dropping `--enforce-eager`). The Mellum model turned out to be graph-capturable out of the box, contrary to "if Step 0 cannot capture" in the task text. Wins: scenario A 714 -> 1003 tokens/s (1.40x, wall 183.5 s -> 130.7 s), single-stream TPOT 29.06 -> 4.89 ms (5.9x).
- Then checked tokens, and the win came with a regression: on C-0003 (6183 input tokens) level 3 diverges from level 0 at generated step 2, while C-0000 (7881 tokens) and the short prompt are 32/32 identical. Level 3 splits the graph at `aiter.unified_attention_with_output`, so prefill no longer runs through the varlen FlashAttention path that commit `5b72cc44` fixed. In `atom/model_ops/attention_mha.py` the sliding-window argument is `window-1` on the decode and varlen-Flash paths but the raw `self.sliding_window` on the paged/persistent-ASM path, so exactly one of them is off by one key. Decision: report point 1 as an *unvalidated candidate* with its measured win and its correctness gap, not as a finished optimization.
- Separately, the HF-side probe from earlier still stands: the reference greedy path reproduces itself at only 313/320 positions on these prompts, so the 90% gate needs a reference whose numerics are pinned down before the remaining agreement gap can be attributed.

### D3 (cont.) - 2026-10-29, 02:00-02:30 Beijing time

- Went to unify the sliding-window convention across ATOM's attention paths, and while checking which path level 3 actually selects found that `use_triton_attn` only depends on window size / head geometry, so level 0 and level 3 take the *same* attention path. That killed my explanation for the token difference and sent me to test the engine's reproducibility instead.
- Cross-level probe: level 0 against itself, level 1 (dynamo-as-is), level 3. Level 0 disagreed with its own second run on C-0003 (2/32), while a third capture matched the first. So the engine, not the compiler, was the variable.
- Repeat-request probe (`scripts/repeat_tokens.py`, 8 identical greedy requests per prompt): 3 distinct token streams on a 10-token prompt with 64 output tokens, at level 3, at level 0, and with the MoE forced onto the aiter kernel via `ATOM_USE_TRITON_MOE=0`. C-0000 is stable 8/8 in every configuration.
- Repeated the whole 5-prompt comparison ten times against the identical HF reference file: agreement ranged 0.250-0.594 (mean 0.469) on a warm server. This is the important result of the day: my earlier "59.1% after the window fix" was one sample, and the level-0-vs-level-3 delta I reported an hour ago is *inside* the engine's own noise.
- Decision: stop treating single-run agreement as a measurement. Ruled out compile level, MoE kernel, and context length. Next suspect is the paged-attention / KV-block-layout path (reduction order depends on the block allocation a request receives). I would rather spend the next block of time making the engine deterministic than chasing another percentage point of agreement - without determinism the 90% gate is not even measurable.
- Corrected `RESULTS.md` in place rather than editing history away, so the wrong reading and the probe that overturned it are both visible.

### D3 (cont.) - 02:30-02:50 Beijing time

- Bisected the nondeterminism instead of guessing. Output lengths 1/2/4/8/12/16/20/24/32/48 are perfectly reproducible over 10 requests each; 64 is not. Across 10 runs at length 64 the common prefix is exactly 55 generated tokens, so the first unstable prediction sits at context length 65.
- So it is a single step that goes bad once the context passes 64 tokens, not diffuse jitter. `context_partition_size` is 128/256 so the attention partition path is not the explanation; KV block size is 16, making 64 = 4 blocks.
- This also explains why C-0000 looked stable and C-0003 did not: the instability only shows when the affected step happens to be near a tie. I had been reading that as evidence that some paths work and others do not - wrong. Recorded that in `RESULTS.md` so the next person does not repeat it.
- Next concrete test, if I pick this up: vary `kv_cache_block_size` and check whether the unstable index moves with it.

### D3 (cont.) - 02:50-03:05 Beijing time

- Ran the block-size discrimination test to check my own "4 blocks = 64 tokens" hypothesis. It failed in an informative way: block size 32 and 64 are unstable from the first decoded token (10 distinct streams / 10 runs), while the default 16 is stable to 64 tokens. So block size changes which attention kernel runs, and my hypothesis is not confirmed - I wrote it up as an open question instead of a cause.
- Useful by-product: default `block_size=16` is the least unstable of the three, so the KV-block axis is a thing to re-probe after any future change rather than a knob to tune for speed.
- Restored the canonical Step 0 server (level 0, eager, bf16 KV, default block size, prefix cache off) so the machine is left in the configuration that `RESULTS.md` describes.

### D3 (cont.) - 03:05-03:25 Beijing time

- Swept prompt length instead of trusting one prompt. 21 / 61 / 130 / 294 / 997 / 2989 tokens with 10 identical requests each. The 61-token prompt stayed bit-identical across 960 decode steps while crossing the alleged 65-token boundary, and the 997 / 2989 prompts diverged at the first or second token. So the "64 threshold" I reported an hour ago was a coincidence on one prompt, not a structural edge - corrected in `RESULTS.md` rather than quietly dropped.
- The graded behaviour (stable at <=300 tokens, immediately divergent at >=1k) points at the split/partition count of the attention reduction growing with context length, which is a much better-fitting hypothesis than KV block boundaries. That is now the stated next probe.
- On block size, definitively: it is not a fixable knob. Changing it swaps the attention kernel (32/64 diverge from the first token) rather than only changing layout, so it stays at the default and any future tuning that touches it has to re-run the determinism probe first.

### D3 (cont.) - 03:25-03:45 Beijing time

- Chased the "kernel never reads the earlier blocks" idea properly instead of dismissing it. The docstring of `pa_decode_gluon` states `max_context_partition_num = ceil(max_context_length / context_partition_size)` while ATOM forces 1 partition of 128 keys for sliding layers, which literally reads as "sliding layers only see the last 128 keys". Designed a test that would settle it: 284-token prompt, where the engine is bit-identical across runs, so any disagreement with HF is systematic rather than noise.
- Result: 64/64 tokens identical to HF, five repetitions, agreement 1.0000. Theory rejected, and the kernel's partition count is a tiling parameter rather than a coverage limit.
- The by-product is the most useful result of the whole investigation: the Mellum port reproduces HF exactly whenever the engine is deterministic. So the integration (weight mapping, 3:1 layer pattern, per-layer RoPE/YaRN, QK-norm, top-8 MoE) is validated, and the ~50% figures were never a correctness measurement.
- Reframed the remaining problem as long-context numerics only, and wrote down the specific suspect: the multi-partition softmax combine in `paged_attention_triton`, whose `exp_sums` / `max_logits` / `temporary_output` buffers are `torch.empty` (uninitialised) and which reduces up to 8 partitions.
