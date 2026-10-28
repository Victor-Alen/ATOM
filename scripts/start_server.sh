#!/bin/bash
# Start (or restart) the single-card ATOM OpenAI server for Mellum 2.1.
#
# Step 0 configuration (the task's "most naive port"):
#   compile level 0, enforce eager, bf16 KV, MTP off, SWA on, prefix cache off.
#
# Optimization-point-1 configuration (piecewise compile + CUDA graph capture):
#   same flags minus --enforce-eager, with --level 3.
#
# Usage:  bash scripts/start_server.sh [LEVEL] [TAG]
#         LEVEL=0  -> step0 (eager, no compilation)
#         LEVEL=3  -> piecewise torch.compile + cudagraph capture
set -u
LEVEL="${1:-0}"
TAG="${2:-level${LEVEL}}"
MODEL=/data/models/Mellum2.1-12B-A2.5B-Thinking
LOG=/workspace/logs/server_${TAG}.log

pkill -f "atom.entrypoints.openai_server" 2>/dev/null
for _ in $(seq 1 60); do
  pgrep -f "atom.entrypoints.openai_server" >/dev/null || break
  sleep 1
done

EXTRA=()
if [ "$LEVEL" = "0" ]; then
  EXTRA+=(--enforce-eager)
fi

cd /workspace/ATOM
nohup python3 -m atom.entrypoints.openai_server \
  --model "$MODEL" \
  --level "$LEVEL" \
  "${EXTRA[@]}" \
  --kv_cache_dtype bf16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.35 \
  --host 127.0.0.1 \
  --server-port 8000 \
  --no-enable_prefix_caching \
  > "$LOG" 2>&1 &
echo "started level=$LEVEL tag=$TAG pid=$! log=$LOG"

for i in $(seq 1 300); do
  if curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "HEALTHY after ${i}s"
    exit 0
  fi
  sleep 1
done
echo "HEALTH TIMEOUT"
exit 1
