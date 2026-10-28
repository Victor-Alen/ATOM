#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ATOM
export PYTHONPATH=/workspace/ATOM
export AITER_LOG_LEVEL=WARNING

exec python3 -m atom.entrypoints.openai_server \
  --model /data/models/Mellum2.1-12B-A2.5B-Thinking \
  --level 0 \
  --enforce-eager \
  --kv_cache_dtype bf16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.35 \
  --host 127.0.0.1 \
  --server-port 8000
