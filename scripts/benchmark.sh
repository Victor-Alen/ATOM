#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-/opt/venv/bin/python3}" "$SCRIPT_DIR/benchmark.py" "$@"
