#!/bin/bash
# Decisive test for "sliding layers only see context_partition_size tokens".
#
# In the 21-294 token range the engine is bit-identical across 10 runs, so any
# disagreement with HF there is a *systematic* implementation difference, not
# run-to-run noise.  If the sliding path really truncates coverage to
# 1 x 128 = 128 tokens, a 290-token prompt must disagree with HF clearly,
# because HF's sliding layers see up to 1024 keys.
set -u
cd /workspace/ATOM
mkdir -p /workspace/correctness

# Build a ~290-token prompt file from the head of C-0003.
python3 - <<'PY'
import json
src = None
with open('/data/datasets/scenario_c_prompts.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 3:
            src = json.loads(line)['prompt']
            break
chars_per_token = len(src) / 6183
cut = int(290 * chars_per_token)
rec = {"id": "SHORT-290", "prompt": src[:cut]}
with open('/workspace/correctness/short290.jsonl', 'w', encoding='utf-8') as f:
    f.write(json.dumps(rec) + "\n")
print("wrote short290.jsonl chars=", cut)
PY

echo "### HF reference (eager, 64 greedy tokens)"
export CUDA_VISIBLE_DEVICES=0
python3 scripts/hf_reference_generate.py \
  --dataset /workspace/correctness/short290.jsonl \
  --output /workspace/correctness/hf_short290_64.jsonl \
  --max-new-tokens 64 --attention eager 2>&1 | grep -E "DONE|prompt_tokens|Error" | tail -3

echo "### ATOM vs HF, 5 repetitions"
for i in 1 2 3 4 5; do
  python3 scripts/compare_atom_hf.py \
    --input /workspace/correctness/hf_short290_64.jsonl \
    --output-dir "/workspace/correctness/short290_rep${i}" \
    --limit 1 --max-tokens 64 > /dev/null 2>&1
  python3 - <<PY
import json
s = json.load(open("/workspace/correctness/short290_rep${i}/summary.json"))
print(f"rep${i}: agreement={s['token_position_agreement']:.4f} matched={s['matched_positions']}/{s['total_positions']} exact={s['exact_sequence_count']}/{s['count']}")
PY
done
echo "### DONE"
