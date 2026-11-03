#!/usr/bin/env bash
set -euo pipefail
export ATOM_MELLUM_HF_ATTN_ROUNDING=1
exec "$(dirname "$0")/start_mellum_norm_hf.sh"
