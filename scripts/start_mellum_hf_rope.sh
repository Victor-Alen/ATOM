#!/usr/bin/env bash
set -euo pipefail
export ATOM_MELLUM_HF_ROPE_INV_FREQ=1
exec "$(dirname "$0")/start_mellum_split_qkv.sh"
