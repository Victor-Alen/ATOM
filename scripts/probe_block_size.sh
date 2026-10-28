#!/bin/bash
# Decisive test: does the first unstable step move when the KV block size
# changes?  If it tracks the block size, the block path is the cause; if it
# stays at context length 65, the 64 threshold belongs to something else.
set -u
cd /workspace/ATOM
MODEL=/data/models/Mellum2.1-12B-A2.5B-Thinking

for BS in 32 64; do
  LOG=/workspace/logs/server_bs${BS}.log
  pkill -f "atom.entrypoints.openai_server" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "atom.entrypoints.openai_server" >/dev/null || break; sleep 1; done
  nohup python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" --level 0 --enforce-eager --kv_cache_dtype bf16 \
    --block-size "$BS" --max-model-len 32768 --gpu-memory-utilization 0.35 \
    --host 127.0.0.1 --server-port 8000 --no-enable_prefix_caching \
    > "$LOG" 2>&1 &
  ok=0
  for i in $(seq 1 300); do
    if curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
    sleep 1
  done
  if [ "$ok" = "0" ]; then
    echo "block-size=$BS FAILED to start:"; tr '\r' '\n' < "$LOG" | tail -12
    continue
  fi
  echo "block-size=$BS healthy"
  python3 - <<PY
import json, urllib.request
URL='http://127.0.0.1:8000/v1/completions'
MODEL='$MODEL'
def post(p, mt):
    req=urllib.request.Request(URL, data=json.dumps({"model":MODEL,"prompt":p,"max_tokens":mt,
        "temperature":0.0,"ignore_eos":True,"stream":False}).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["token_ids"]
prompt="def add(a, b):\n    return"
runs=[post(prompt,96) for _ in range(10)]
n=len(runs[0])
first=next((j for j in range(n) if len({r[j] for r in runs})>1), None)
print(f"block-size=$BS: distinct={len({tuple(r) for r in runs})}/10 first_unstable_generated_index={first} "
      f"context_len_at_that_step={None if first is None else 10+first}")
PY
done
echo "### DONE"
