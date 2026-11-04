#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ATOM-general-norm
while ps --ppid 895993 -o args= | grep -q '^python3 -u scripts/compare_atom_hf.py'; do sleep 3; done
test -f /workspace/logs/step0_causal_boundary/exact_sliding_service/C10_64/summary.json
root=/workspace/logs/step0_causal_boundary/acceptance_a20_c200_v1
mkdir -p "$root"
old=$(cat /workspace/logs/step0_causal_boundary/exact_sliding_service/server.pid)
kill -TERM "$old"
for i in $(seq 1 60); do
 state=$(ps -p "$old" -o stat= || true)
 if test -z "$state" || [[ "$state" == Z* ]]; then break; fi
 sleep 1
done
state=$(ps -p "$old" -o stat= || true)
if test -n "$state" && [[ "$state" != Z* ]]; then echo OLD_SERVER_STILL_LIVE; exit 1; fi
export ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=1 AITER_LOG_LEVEL=WARNING MELLUM_SERVER_PORT=8005
bash scripts/start_mellum_hf_arithmetic.sh > "$root/server.log" 2>&1 &
server=$!;echo "$server" > "$root/server.pid"
for i in $(seq 1 150); do
 kill -0 "$server"
 if curl -fsS http://127.0.0.1:8005/health >/dev/null 2>&1; then break; fi
 sleep 2
done
python3 -u scripts/write_acceptance_manifest.py --root "$root" --pid "$server"
rocm-smi --showmeminfo vram > "$root/gpu_running.txt"
python3 -u scripts/compare_atom_hf.py --input /workspace/correctness/hf_scenario_a_20_eager.jsonl --output-dir "$root/A20_64" --url http://127.0.0.1:8005/v1/completions --max-tokens 64
python3 -u scripts/compare_atom_hf.py --input /workspace/correctness/hf_scenario_b_20_eager.jsonl --output-dir "$root/B20_64" --url http://127.0.0.1:8005/v1/completions --max-tokens 64
python3 -u scripts/compare_atom_hf.py --input /workspace/correctness/hf_scenario_c_200_eager.jsonl --output-dir "$root/C200_64" --url http://127.0.0.1:8005/v1/completions --max-tokens 64
echo FORMAL_ACCEPTANCE_RUN_DONE
