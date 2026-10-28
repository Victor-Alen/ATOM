#!/bin/bash
# Test whether the greedy nondeterminism comes from the Triton fused MoE's
# atomic accumulation: same level-0 server, MoE implementation forced to the
# aiter path via ATOM_USE_TRITON_MOE=0.
set -u
cd /workspace/ATOM
MODEL=/data/models/Mellum2.1-12B-A2.5B-Thinking
LOG=/workspace/logs/server_moe_aiter.log

pkill -f "atom.entrypoints.openai_server" 2>/dev/null
for _ in $(seq 1 60); do pgrep -f "atom.entrypoints.openai_server" >/dev/null || break; sleep 1; done

ATOM_USE_TRITON_MOE=0 nohup python3 -m atom.entrypoints.openai_server \
  --model "$MODEL" --level 0 --enforce-eager --kv_cache_dtype bf16 \
  --max-model-len 32768 --gpu-memory-utilization 0.35 \
  --host 127.0.0.1 --server-port 8000 --no-enable_prefix_caching \
  > "$LOG" 2>&1 &
echo "started aiter-MoE server pid=$!"

ok=0
for i in $(seq 1 300); do
  if curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; echo "HEALTHY after ${i}s"; break; fi
  sleep 1
done
if [ "$ok" = "0" ]; then
  echo "START FAILED - log tail:"
  tr '\r' '\n' < "$LOG" | tail -20
  exit 1
fi

python3 scripts/repeat_tokens.py --tag "aiter-moe" --repeat 8 --include-short \
  --out /workspace/bench/repeat_moe_aiter.json 2>&1 | grep -E "distinct"
echo "### DONE"
