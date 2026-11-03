#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export AITER_LOG_LEVEL=WARNING
export ATOM_MELLUM_MOE_COMBINE="${ATOM_MELLUM_MOE_COMBINE:-ordered}"

exec python3 -m atom.entrypoints.openai_server \
  --model /data/models/Mellum2.1-12B-A2.5B-Thinking \
  --level 0 \
  --enforce-eager \
  --kv_cache_dtype bf16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --host 127.0.0.1 \
  --server-port "${MELLUM_SERVER_PORT:-8003}" \
  --no-enable_prefix_caching
