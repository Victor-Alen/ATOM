#!/bin/bash
# Distinguish "level 3 changed the attention code path" from "inductor picked
# numerically different kernels" by measuring token stability at each
# compilation level, plus the determinism of each level against itself.
set -u
cd /workspace/ATOM

echo "### level 3, run 2 (server already running at level 3)"
python3 scripts/capture_tokens.py --out /workspace/bench/level3_tokens_run2.json 2>&1 | tail -1

for LEVEL in 0 1 3; do
  echo "### restarting at level $LEVEL"
  bash scripts/start_server.sh "$LEVEL" "probe$LEVEL" || { echo "level $LEVEL failed to start"; continue; }
  python3 scripts/capture_tokens.py --out "/workspace/bench/level${LEVEL}_tokens_run1.json" 2>&1 | tail -1
  python3 scripts/capture_tokens.py --out "/workspace/bench/level${LEVEL}_tokens_run2.json" 2>&1 | tail -1
done

echo "### comparison"
python3 - <<'PY'
import json, itertools, os
def load(name):
    path = f"/workspace/bench/{name}.json"
    if not os.path.exists(path):
        return None
    return {r["id"]: r["token_ids"] for r in json.load(open(path))}

runs = {n: load(n) for n in [
    "step0_tokens",
    "level0_tokens_run1", "level0_tokens_run2",
    "level1_tokens_run1", "level1_tokens_run2",
    "level3_tokens", "level3_tokens_run2",
]}
pairs = [
    ("level0_tokens_run1", "level0_tokens_run2"),   # level 0 self-determinism
    ("level3_tokens", "level3_tokens_run2"),        # level 3 self-determinism
    ("level0_tokens_run1", "level1_tokens_run1"),   # does compile-as-is change tokens?
    ("level0_tokens_run1", "level3_tokens_run2"),   # does inductor+cudagraph change tokens?
    ("step0_tokens", "level0_tokens_run1"),         # cross-restart stability of level 0
]
for left, right in pairs:
    a, b = runs.get(left), runs.get(right)
    if not a or not b:
        print(f"{left} vs {right}: MISSING")
        continue
    print(f"{left} vs {right}:")
    for key in a:
        x, y = a[key], b[key]
        same = sum(1 for i, j in zip(x, y) if i == j)
        print(f"   {key}: {same}/{len(x)} identical={x == y}")
PY
