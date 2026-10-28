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
