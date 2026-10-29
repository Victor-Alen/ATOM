#!/bin/bash
# Does the prefill sliding-window off-by-one actually change the output?
#
# Problem: with the real window (1024) the sliding mask never activates on short
# prompts, and at contexts >1024 the engine is not reproducible, so the fix
# could not be validated either way.
#
# Trick: copy the config with sliding_window = 16. A 100-token prompt then has
# 100 > 16 so the window is active, while 100 tokens remains inside the
# reproducible range. Same HF reference, two ATOM builds (fix present / reverted).
set -u
cd /workspace/ATOM
SRC=/data/models/Mellum2.1-12B-A2.5B-Thinking
DST=/workspace/models/sw16
mkdir -p "$DST" /workspace/correctness
for f in "$SRC"/*; do
  base=$(basename "$f")
  [ "$base" = "config.json" ] && continue
  ln -sfn "$f" "$DST/$base"
done
python3 - <<'PY'
import json
src = "/data/models/Mellum2.1-12B-A2.5B-Thinking/config.json"
dst = "/workspace/models/sw16/config.json"
cfg = json.load(open(src))
cfg["sliding_window"] = 16
json.dump(cfg, open(dst, "w"), indent=2)
print("sw16 config: sliding_window =", cfg["sliding_window"], "layers:", len(cfg["layer_types"]))
PY

# 100-token prompts from three different scenario-C prompts, to diversify ties.
python3 - <<'PY'
import json
want = {0: "SW16-A", 3: "SW16-B", 4: "SW16-C"}
recs = []
with open('/data/datasets/scenario_c_prompts.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i not in want:
            if i > 4:
                break
            continue
        p = json.loads(line)['prompt']
        chars_per_token = len(p) / 6183
        recs.append({"id": want[i], "prompt": p[: int(100 * chars_per_token)]})
with open('/workspace/correctness/sw16_prompts.jsonl', 'w', encoding='utf-8') as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print("prompts:", [r["id"] for r in recs])
PY

# HF reference under the same window=16 config.
sed 's#^MODEL_PATH = .*#MODEL_PATH = "/workspace/models/sw16"#' \
  scripts/hf_reference_generate.py > scripts/_hf_ref_sw16.py
echo "### HF reference (window=16, eager, 64 tokens)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/_hf_ref_sw16.py \
  --dataset /workspace/correctness/sw16_prompts.jsonl \
  --output /workspace/correctness/hf_sw16_64.jsonl \
  --max-new-tokens 64 --attention eager 2>&1 | grep -E "DONE|Error|Traceback" | tail -4

start_atom () {
  pkill -f "atom.entrypoints.openai_server" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "atom.entrypoints.openai_server" >/dev/null || break; sleep 1; done
  nohup python3 -m atom.entrypoints.openai_server \
    --model "$DST" --level 0 --enforce-eager --kv_cache_dtype bf16 \
    --max-model-len 32768 --gpu-memory-utilization 0.35 \
    --host 127.0.0.1 --server-port 8000 --no-enable_prefix_caching \
    > /workspace/logs/server_sw16.log 2>&1 &
  for _ in $(seq 1 300); do
    curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "healthy"; return 0; }
    sleep 1
  done
  echo "START FAILED"; tr '\r' '\n' < /workspace/logs/server_sw16.log | tail -15; return 1
}

measure () {
  local tag=$1
  for i in 1 2 3; do
    python3 scripts/compare_atom_hf.py \
      --input /workspace/correctness/hf_sw16_64.jsonl \
      --output-dir "/workspace/correctness/sw16_${tag}_${i}" \
      --max-tokens 64 > /dev/null 2>&1
    python3 - <<PY
import json
s = json.load(open("/workspace/correctness/sw16_${tag}_${i}/summary.json"))
print(f"${tag} run${i}: agreement={s['token_position_agreement']:.4f} matched={s['matched_positions']}/{s['total_positions']} exact={s['exact_sequence_count']}/{s['count']}")
PY
  done
}

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
git diff --stat atom/model_ops/attention_mha.py | tail -1
echo "### restoring canonical step0 server"
bash scripts/start_server.sh 0 step0final 2>&1 | tail -1
echo "### DONE"
