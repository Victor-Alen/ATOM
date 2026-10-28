#!/bin/bash
# Step 0 baseline: run the three task scenarios back to back against the
# currently running single-card ATOM server (eager, level 0, bf16 KV,
# MTP off, SWA on, prefix caching off = the task's "most naive port").
set -u
cd /workspace/ATOM
mkdir -p /workspace/bench
STAMP=$(date +%Y%m%d-%H%M%S)
echo "STAMP=$STAMP"
echo "server: $(pgrep -af 'atom.entrypoints.openai_server' | head -1)"

for SC in A B C; do
  echo "=== scenario $SC start $(date -Is) ==="
  case $SC in
    A) ARGS="--number 512 --parallel 32";;
    B) ARGS="--number 50 --parallel 1";;
    C) ARGS="--number 200 --rate 0.3";;
  esac
  python3 scripts/bench_scenarios.py --scenario $SC $ARGS \
    --out /workspace/bench/step0_${SC}_${STAMP}.json
  echo "=== scenario $SC done $(date -Is) ==="
done
echo "ALL DONE $(date -Is)"
