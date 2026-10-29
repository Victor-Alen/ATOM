#!/bin/bash
# Re-run the window=16 ablation with the correct --model argument
# (ATOM returns HTTP 400 when the request model name differs from the loaded
# path, which is what silently killed the first attempt).
set -u
cd /workspace/ATOM
MODEL=/workspace/models/sw16
HF=/workspace/correctness/hf_sw16_64.jsonl

start_atom () {
  pkill -f "atom.entrypoints.openai_server" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "atom.entrypoints.openai_server" >/dev/null || break; sleep 1; done
  nohup python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" --level 0 --enforce-eager --kv_cache_dtype bf16 \
    --max-model-len 32768 --gpu-memory-utilization 0.35 \
    --host 127.0.0.1 --server-port 8000 --no-enable_prefix_caching \
    > /workspace/logs/server_sw16.log 2>&1 &
  for _ in $(seq 1 300); do
    curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo healthy; return 0; }
    sleep 1
  done
  echo "START FAILED"; tr '\r' '\n' < /workspace/logs/server_sw16.log | tail -12; return 1
}

measure () {
  local tag=$1
  for i in 1 2 3; do
    rm -rf "/workspace/correctness/sw16_${tag}_${i}"
    python3 scripts/compare_atom_hf.py \
      --input "$HF" --model "$MODEL" \
      --output-dir "/workspace/correctness/sw16_${tag}_${i}" \
      --max-tokens 64 2>&1 | tail -3
    python3 - <<PY
import json
try:
    s = json.load(open("/workspace/correctness/sw16_${tag}_${i}/summary.json"))
    print(f"${tag} run${i}: agreement={s['token_position_agreement']:.4f} "
          f"matched={s['matched_positions']}/{s['total_positions']} exact={s['exact_sequence_count']}/{s['count']}")
except FileNotFoundError:
    print("${tag} run${i}: NO SUMMARY")
PY
  done
}

git status --short atom/model_ops/attention_mha.py
echo "### ATOM with the fix (prefill window - 1)"
start_atom || exit 1
measure fixed

echo "### ATOM with the fix reverted (prefill window raw)"
cp atom/model_ops/attention_mha.py /tmp/attention_mha_fixed.py
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/attention_mha.py')
s = p.read_text()
old = """        sliding_window = (
            (self.sliding_window - 1, 0, 0)
            if self.sliding_window > 0
            else (-1, -1, 0)
        )"""
new = """        sliding_window = (
            (self.sliding_window, 0, 0) if self.sliding_window > 0 else (-1, -1, 0)
        )"""
assert old in s, "fix snippet not found"
p.write_text(s.replace(old, new))
print("reverted prefill sliding window to raw")
PY
start_atom || { cp /tmp/attention_mha_fixed.py atom/model_ops/attention_mha.py; exit 1; }
measure reverted

cp /tmp/attention_mha_fixed.py atom/model_ops/attention_mha.py
echo "### restored fix; git status:"
git status --short atom/model_ops/attention_mha.py
echo "### restoring canonical step0 server"
bash scripts/start_server.sh 0 step0final 2>&1 | tail -1
echo "### DONE"
