#!/bin/bash
# Step 0 gate: run the engine's reference path and compare against HF.
# Requires a GPU with >= 28 GiB free for short prompts, >= 45 GiB for the 7881 case.
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL=/data/models/Mellum2.1-12B-A2.5B-Thinking
DEV=${1:-0}
UTIL=${2:-0.15}
PORT=8001
LOG=/workspace/logs/step0_gate_server.log

echo "[gate] starting reference server on GPU $DEV (util $UTIL)"
HIP_VISIBLE_DEVICES=$DEV ATOM_REFERENCE_MODE=1 ATOM_REFERENCE_RECOMPUTE=1 \
  ATOM_REFERENCE_MODEL_DIR=$MODEL \
  python3 -m atom.entrypoints.openai_server --model $MODEL \
  --max-model-len 32768 --gpu-memory-utilization $UTIL --max-num-seqs 1 \
  --host 127.0.0.1 --server-port $PORT --no-enable_prefix_caching > "$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "[gate] server healthy"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[gate] server died, see $LOG"; tail -5 "$LOG"; exit 1
  fi
  sleep 5
done

echo "[gate] short check (1024 prompt tokens, 8 greedy steps)"
python3 scripts/step0_gate.py --port $PORT --tokens 1024 --max-tokens 8 || true

echo "[gate] full check (7881 prompt tokens, 4 greedy steps)"
python3 scripts/step0_gate.py --port $PORT --tokens 7881 --max-tokens 4 || true

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "[gate] done"
