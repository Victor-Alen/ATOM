#!/usr/bin/env bash
set -euo pipefail
export ATOM_MELLUM_HF_NORM_PREFILL=1
exec "$(dirname "$0")/start_mellum_hf_prefill990.sh"
