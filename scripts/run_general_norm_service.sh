#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ATOM-general-norm
logs=/workspace/logs/step0_causal_boundary/general_norm_service
mkdir -p "$logs"
old=$(cat /workspace/logs/step0_causal_boundary/server.pid)
kill -TERM "$old"
for i in $(seq 1 60); do
 state=$(ps -p "$old" -o stat= || true)
 if test -z "$state" || [[ "$state" == Z* ]]; then break; fi
 sleep 1
done
state=$(ps -p "$old" -o stat= || true)
if test -n "$state" && [[ "$state" != Z* ]]; then echo OLD_SERVER_STILL_LIVE; exit 1; fi
unset ATOM_CAUSAL_DIR ATOM_STEP_TRACE_DIR ATOM_LAYER_TRACE_DIR
export ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=1 AITER_LOG_LEVEL=WARNING MELLUM_SERVER_PORT=8005
bash scripts/start_mellum_ck_profile.sh > "$logs/server.log" 2>&1 &
server=$!
echo "$server" > "$logs/server.pid"
for i in $(seq 1 150); do
 kill -0 "$server"
 if curl -fsS http://127.0.0.1:8005/health > /dev/null 2>&1; then break; fi
 sleep 2
done
rocm-smi --showmeminfo vram > "$logs/gpu_running.txt"
for scenario in A C; do
 python3 -u scripts/compare_atom_hf.py --input "/workspace/logs/step0_continuation/ck_hf_profile_prod/fresh_hf/${scenario}10_64.jsonl" --output-dir "$logs/${scenario}10_64" --url http://127.0.0.1:8005/v1/completions --max-tokens 64 --limit 10
done
echo GENERAL_NORM_AC10_DONE
