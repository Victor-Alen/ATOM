#!/usr/bin/env bash
set -euo pipefail
: "${HIP_VISIBLE_DEVICES:?Set the allocated runtime GPU explicitly}"
export ATOM_MELLUM_BF16_HF_ATTENTION=1
export ATOM_MELLUM_HF_GEMM_MOE=1
export ATOM_MELLUM_EXACT_NORM=1
export ATOM_MELLUM_EXACT_LINEAR=1
export ATOM_MELLUM_EXACT_ROPE=1
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0
unset ATOM_REFERENCE_MODE ATOM_PROD_MOE ATOM_EAGER_ATTN
unset ATOM_CAUSAL_DIR ATOM_LAYER_TRACE_DIR ATOM_STEP_TRACE_DIR
echo 'Mellum production: native paged KV; HF-order GPU arithmetic; grouped/batched GEMM experts'
exec "$(dirname "$0")/start_mellum_ck_profile.sh"
