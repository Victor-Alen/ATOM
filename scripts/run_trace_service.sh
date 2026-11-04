#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ATOM-general-norm
root=/workspace/logs/step0_causal_boundary/current_A0002_step3
mkdir -p "$root/atom"
old=$(cat /workspace/logs/step0_causal_boundary/exact_rope_service/server.pid)
kill -TERM "$old"
for i in $(seq 1 60); do
 state=$(ps -p "$old" -o stat= || true)
 if test -z "$state" || [[ "$state" == Z* ]]; then break; fi
 sleep 1
done
state=$(ps -p "$old" -o stat= || true)
if test -n "$state" && [[ "$state" != Z* ]]; then exit 1; fi
export ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=1 AITER_LOG_LEVEL=WARNING MELLUM_SERVER_PORT=8005
export ATOM_MELLUM_BF16_HF_ATTENTION=1 ATOM_MELLUM_HF_GEMM_MOE=1 ATOM_MELLUM_EXACT_NORM=1 ATOM_MELLUM_EXACT_LINEAR=1 ATOM_MELLUM_EXACT_ROPE=1
export ATOM_LAYER_TRACE_DIR="$root/atom" ATOM_STEP_TRACE_DIR="$root/atom" ATOM_STEP_TRACE_STEP=3 ATOM_STEP_TRACE_L0_DETAIL=1
bash scripts/start_mellum_ck_profile.sh > "$root/server.log" 2>&1 &
server=$!;echo "$server" > "$root/server.pid"
for i in $(seq 1 150); do
 kill -0 "$server"
 if curl -fsS http://127.0.0.1:8005/health >/dev/null 2>&1; then break; fi
 sleep 2
done
touch "$root/atom/ARMED"
python3 -u scripts/request_trace.py --root "$root"
