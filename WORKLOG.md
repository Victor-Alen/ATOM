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
