#!/usr/bin/env bash
set -euo pipefail
export CK_DIR=/workspace/ck-mellum-up-fp64
export AITER_META_DIR=/workspace/aiter-mellum-norm-hf
export AITER_JIT_DIR=/workspace/aiter-mellum-norm-hf-rne-jit
export OPUS_FP32_to_BF16_DEFAULT=2
export CK_MELLUM_STAGE1_MODE=1
export CK_MELLUM_STAGE2_ROUND_WEIGHT=1
export ATOM_MELLUM_MOE_COMBINE=ordered
export ATOM_MELLUM_MOE_BF16_ROUTER_WEIGHTS=1
unset ATOM_STEP_TRACE_DIR ATOM_LAYER_TRACE_DIR ATOM_NORM_HF_BF16 ATOM_REFERENCE_MODE ATOM_PROD_MOE ATOM_MOE_HF_ORDER
printf 'AITER source: '; git -C "$AITER_META_DIR" rev-parse HEAD
printf 'CK source: '; git -C "$CK_DIR" rev-parse HEAD
sha256sum "$AITER_JIT_DIR/module_rmsnorm_quant.so"
exec "$(dirname "$0")/start_mellum_ordered.sh"
