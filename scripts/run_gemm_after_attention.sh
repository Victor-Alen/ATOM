#!/usr/bin/env bash
set -euo pipefail
while ps --ppid 883601 -o args= | grep -q '^python3 -u scripts/compare_atom_hf.py'; do sleep 3; done
test -f /workspace/logs/step0_causal_boundary/general_attention_layout_service/C10_64/summary.json
cd /workspace/ATOM-general-norm
export CANDIDATE_LOGDIR=/workspace/logs/step0_causal_boundary/general_gemm_service
export PREVIOUS_SERVER_PID_FILE=/workspace/logs/step0_causal_boundary/general_attention_layout_service/server.pid
export ATOM_MELLUM_BF16_HF_ATTENTION=1 ATOM_MELLUM_HF_GEMM_MOE=1
exec bash scripts/run_general_norm_service.sh
