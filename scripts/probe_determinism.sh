#!/bin/bash
# Where does the greedy nondeterminism live?
#   - short prompt (10 in / 64 out) stable + long prompt unstable -> attention/KV
#   - both unstable                                              -> MoE reduction
# Run at level 3 (currently up) and at level 0 (eager).
set -u
cd /workspace/ATOM

echo "### level 3 (already running)"
python3 scripts/repeat_tokens.py --tag "level3" --repeat 8 --include-short \
  --out /workspace/bench/repeat_level3_short.json 2>&1 | grep -E "distinct"

echo "### restart at level 0"
bash scripts/start_server.sh 0 probe0det || echo "level 0 failed"
python3 scripts/repeat_tokens.py --tag "level0" --repeat 8 --include-short \
  --out /workspace/bench/repeat_level0_short.json 2>&1 | grep -E "distinct"

echo "### DONE"
