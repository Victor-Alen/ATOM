#!/usr/bin/env bash
set -euo pipefail
export CK_DIR=/workspace/ck-mellum-up-fp64
export AITER_JIT_DIR=/workspace/ck-mellum-up-fp64-jit
export CK_MELLUM_STAGE1_MODE=1
export CK_MELLUM_STAGE2_ROUND_WEIGHT=1
export ATOM_MELLUM_MOE_COMBINE=ordered
module="$AITER_JIT_DIR/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so"
test -f "$module"
echo "Mellum BF16 V3 up projection: independent FP32 chunks and FP64 merge"
git -C "$CK_DIR" rev-parse HEAD
sha256sum "$module"
exec "$(dirname "$0")/start_mellum_ordered.sh"
