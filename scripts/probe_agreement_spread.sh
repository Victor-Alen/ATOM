#!/bin/bash
# How much of the ATOM-vs-HF agreement number is measurement noise?
# Same server, same prompts, same HF reference file, five repetitions.
set -u
cd /workspace/ATOM

bash scripts/start_server.sh 0 step0clean || { echo "restart failed"; exit 1; }

for i in 1 2 3 4 5; do
  python3 scripts/compare_atom_hf.py \
    --input /workspace/correctness/hf_scenario_c_200_eager.jsonl \
    --output-dir "/workspace/correctness/repeat_agreement_${i}" \
    --limit 5 --max-tokens 64 > /dev/null 2>&1
  python3 - <<PY
import json
s = json.load(open("/workspace/correctness/repeat_agreement_${i}/summary.json"))
print(f"run ${i}: agreement={s['token_position_agreement']:.4f} "
      f"exact={s['exact_sequence_count']}/{s['count']} matched={s['matched_positions']}/{s['total_positions']}")
PY
done
echo "### DONE"
