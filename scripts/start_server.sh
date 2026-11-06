#!/usr/bin/env bash
# Three formal points: Attention, compile/Graph, MTP K=1. See README-CSIG.md.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-/opt/venv/bin/python3}" "$SCRIPT_DIR/start_server.py" "$@"
