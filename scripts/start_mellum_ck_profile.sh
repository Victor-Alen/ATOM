#!/usr/bin/env bash
set -euo pipefail
export CK_DIR=/workspace/ck-mellum-hf-profile
export AITER_META_DIR=/workspace/aiter-mellum-norm-hf
export AITER_JIT_DIR=/workspace/ck-mellum-hf-profile-jit
export PYTHONPATH="/workspace/aiter-mellum-norm-hf${PYTHONPATH:+:$PYTHONPATH}"
export CK_MELLUM_STAGE1_MODE=1
export CK_MELLUM_STAGE2_ROUND_WEIGHT=1
export CK_MELLUM_STAGE1_KORDER_PLAN=1
export ATOM_MELLUM_MOE_HF_KORDER=1
export ATOM_MELLUM_MOE_COMBINE=ordered
export ATOM_MELLUM_MOE_BF16_ROUTER_WEIGHTS=1
export ATOM_MELLUM_HF_NORM_PREFILL=1
export ATOM_MELLUM_HF_ATTN_PREFILL_ROUNDING=1
export ATOM_MELLUM_HF_ATTN_ROUNDING=1
export ATOM_MELLUM_SPLIT_QKV_PREFILL=1
export AITER_MELLUM_HF_ROPE_INV_FREQ=1
export OPUS_FP32_to_BF16_DEFAULT=2
unset ATOM_NORM_HF_BF16 ATOM_REFERENCE_MODE ATOM_PROD_MOE ATOM_MOE_HF_ORDER
module="$AITER_JIT_DIR/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so"
test -f "$module"
git -C "$CK_DIR" rev-parse HEAD
sha256sum "$module" "$AITER_JIT_DIR/module_rmsnorm_quant.so"
exec "$(dirname "$0")/start_mellum_ordered.sh"
