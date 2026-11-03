#!/usr/bin/env bash
set -euo pipefail
variant="${1:-stage1}"
case "$variant" in
  stage1)
    export CK_DIR=/workspace/ck-mellum-hf-round
    export AITER_JIT_DIR=/workspace/ck-mellum-hf-jit
    export CK_MELLUM_STAGE2_ROUND_WEIGHT=0
    ;;
  stage12)
    export CK_DIR=/workspace/ck-mellum-hf-stage12
    export AITER_JIT_DIR=/workspace/ck-mellum-hf-stage12-jit
    export CK_MELLUM_STAGE2_ROUND_WEIGHT=1
    ;;
  *) echo "Expected stage1 or stage12" >&2; exit 2 ;;
esac
export CK_MELLUM_STAGE1_MODE=1
export ATOM_MELLUM_MOE_COMBINE=ordered
module="$AITER_JIT_DIR/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so"
test -f "$module"
git -C "$CK_DIR" rev-parse HEAD
sha256sum "$module"
exec "$(dirname "$0")/start_mellum_ordered.sh"
