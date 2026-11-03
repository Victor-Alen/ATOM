#!/usr/bin/env bash
set -euo pipefail
export ATOM_MELLUM_SPLIT_QKV_PREFILL=1
exec "$(dirname "$0")/start_mellum_hf_attn_rounding.sh"
