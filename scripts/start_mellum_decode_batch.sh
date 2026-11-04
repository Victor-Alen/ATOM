#!/usr/bin/env bash
set -euo pipefail
export ATOM_MELLUM_BATCH_DECODE_MOE=1
export MELLUM_SERVER_PORT="${MELLUM_SERVER_PORT:-8018}"
export MELLUM_ENGINE_PORT="${MELLUM_ENGINE_PORT:-18018}"
exec "$(dirname "$0")/start_mellum_cached.sh"
