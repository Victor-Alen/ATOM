#!/usr/bin/env bash
set -euo pipefail
export AITER_MELLUM_HF_ROPE_INV_FREQ=1
export PYTHONPATH="/workspace/aiter-mellum-norm-hf${PYTHONPATH:+:$PYTHONPATH}"
exec "$(dirname "$0")/start_mellum_split_qkv.sh"
