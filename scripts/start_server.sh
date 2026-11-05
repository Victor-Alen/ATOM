#!/usr/bin/env bash
# Default: fast native candidate, MTP disabled. See START_SERVER.md.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-/opt/venv/bin/python3}" "$SCRIPT_DIR/start_server.py" "$@"
