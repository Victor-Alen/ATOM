# Step 0 overnight runbook (2026-11-02)

Current state: **not passed**. Best measured agreement: A 0.3516-0.3766, B 1.0000,
C 0.5816 (40 of 200 prompts, the rest exceed the 12k single-shot limit).

## What is proven

- weights, embedding, layer-0 input LN and post-rope k are bit-identical to HF (0.000000);
  post-rope q differs by 0.0078 (sub-ulp, operation order).
- The standalone probe (my reference code + HF in one process) reports
  `d_attn = d_mlp = d_block = 0.000000` for the layers it completes.
- The same prompt sent repeatedly to the pristine engine returns different answers
  (first request anomalous), and requests leak state into each other.

## What is not solved

1. The engine process and a plain torch/HF process resolve different GEMM kernels for the
   same shapes. Pinning `TORCH_BLAS_PREFER_HIPBLASLT=0` on both sides makes agreement
   worse, not better, and flips individual prompts (A-0007: 16/16 in one config, 1/64 in another).
2. Because of (1) the 90% gate cannot be evaluated meaningfully until both sides provably
   share kernels.

## Next experiments, in order

1. **Force rocBLAS in both**: verify what `TORCH_BLAS_PREFER_HIPBLASLT` actually does in
   this torch build (`torch.backends.cuda.preferred_blas_library()`), then pin the same
   library on both sides and re-measure A-0006 (known 8/8 target) plus A-0000.
2. **In-process HF reference**: load HF's `MellumModel` inside the engine process and use it
   as the reference. If it reproduces standalone HF, plumbing is the issue; if not, the gate
   needs an explicit environment pin.
3. **Make the reference path independent of engine state**: feed it only the captured prompt
   (the `llm_engine` capture file already exists), reset `_HIST`/`_KV` per request, and
   ignore the engine's padded input buffer entirely.
4. **Warm-up rule**: always send one throwaway request before measuring (the first request
   after start is anomalous).
5. Only after 1-4: run the full 200-prompt gate.

## How to run things

- Reference server: `/tmp/start_ref_dbg.sh` (env: ATOM_REFERENCE_MODE=1, RECOMPUTE=1,
  MOE_DET=1, DEBUG=1, DUMPL=/workspace/logs/dumpl), port 8001, GPU0.
- Pristine server for reproduction: `/tmp/start_pristine.sh`, port 8003, GPU0.
- Gate runner: `scripts/gate_run.py` (A against fresh refs, B and C against recorded refs).
- Single prompt compare: `scripts/one_prompt.py <PID>`.
- Latched layer dumps for bisection: `/workspace/logs/dumpl/layers_{0,1}.pt`, compare with
  `scripts/layers_cmp.py` against `/workspace/logs/hf_hs.pt`.

## Repository state

Both `/workspace/ATOM-tritonmoe` (my branch `dsh/moe-determinism`) and the shared
`/workspace/ATOM` RESULTS.md are committed and in sync (round 40).
