# WORKLOG

## D1 - 2026-10-28

- Confirmed the assigned host has two AMD Instinct MI308X GPUs and the frozen Mellum 2.1 weights under `/data/models`. Kept `/app/ATOM` unchanged and used `/workspace/ATOM` at upstream commit `26f23a0b` for the integration work.
- Registered the `mellum` config and `MellumForCausalLM` architecture. Added a Mellum model based on the Qwen3-MoE weight layout, with per-layer full/sliding attention and per-attention-type RoPE/YaRN configuration.
- Started a single-engine ATOM OpenAI server in eager mode at compilation level 0 with bf16 KV cache and a 32,768-token maximum length. Verified `/health` and observed HTTP 200 completion requests with 5, 990, and 7,881 input tokens.
- Reviewed the initial integration and confirmed it already calls ATOM `FusedMoE`; a generic "replace Python MoE loop with fused MoE" item is therefore not a valid later optimization for this Step 0.
- Correctness is not yet certified: the 200-prompt first-64-token HF comparison, separate 16k check, and HumanEval comparison are still pending. No formal scenario A/B/C baseline result has been recorded.
