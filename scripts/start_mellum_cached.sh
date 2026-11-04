#!/usr/bin/env bash
set -euo pipefail
: "${HIP_VISIBLE_DEVICES:?Set the allocated runtime GPU explicitly}"
export ATOM_MELLUM_REQUEST_GEMM=1
export ATOM_MELLUM_CACHE_METADATA=1
export MELLUM_SERVER_PORT="${MELLUM_SERVER_PORT:-8015}"
export MELLUM_ENGINE_PORT="${MELLUM_ENGINE_PORT:-18015}"
exec "$(dirname "$0")/start_mellum_hf_arithmetic.sh"
