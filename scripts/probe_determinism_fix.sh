#!/bin/bash
# Determinism hunt: try each plausible fix for the decode nondeterminism and
# use the repeat test as the judge. Judge = distinct token streams over 8
# identical greedy requests on 3 prompts. Baseline today: short64=3, C-0003=2-3.
#
# Each variant: patch -> restart -> repeat -> git checkout (restore).
set -u
cd /workspace/ATOM
FILE=atom/model_ops/attention_mha.py
OUT=/workspace/bench

judge () {
  local tag=$1
  if ! bash scripts/start_server.sh 0 "det_${tag}" >/dev/null 2>&1; then
    echo "  ${tag}: START FAILED"
    return 1
  fi
  python3 scripts/repeat_tokens.py --tag "$tag" --repeat 8 --include-short \
    --out "$OUT/det_${tag}.json" 2>&1 | grep -E "distinct" | sed 's/^/  /'
}

echo "=== V1: sliding layers use context_partition_size 256 instead of 128"
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/attention_mha.py'); s = p.read_text()
old = "                context_partition_size = 128"
assert old in s, "V1 anchor missing"
p.write_text(s.replace(old, "                context_partition_size = 256"))
PY
judge v1_part256
git checkout -- $FILE

echo "=== V2: force max_context_partition_num = 1 for every layer"
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/attention_mha.py'); s = p.read_text()
old = """            context_partition_size = 256
            if self.sliding_window > 0:
                max_context_partition_num = 1
                context_partition_size = 128
"""
assert old in s, "V2 anchor missing"
new = old + "            max_context_partition_num = 1\n"
p.write_text(s.replace(old, new))
PY
judge v2_onepart
git checkout -- $FILE

echo "=== V3: initialise the partition buffers with zeros instead of torch.empty"
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/attention_mha.py'); s = p.read_text()
pairs = [
 ("""            exp_sums = torch.empty(
                intermediate_shape, dtype=torch.float32, device=q.device
            )""",
  """            exp_sums = torch.zeros(
                intermediate_shape, dtype=torch.float32, device=q.device
            )"""),
 ("""            max_logits = torch.empty(
                intermediate_shape, dtype=torch.float32, device=q.device
            )""",
  """            max_logits = torch.full(
                intermediate_shape, float("-inf"), dtype=torch.float32, device=q.device
            )"""),
 ("""            temporary_output = torch.empty(
                *intermediate_shape,
                head_size,
                dtype=q.dtype,
                device=q.device,
            )""",
  """            temporary_output = torch.zeros(
                *intermediate_shape,
                head_size,
                dtype=q.dtype,
                device=q.device,
            )"""),
]
for old, new in pairs:
    assert old in s, f"V3 anchor missing: {old.splitlines()[0]}"
    s = s.replace(old, new)
p.write_text(s)
PY
judge v3_zerobuf
git checkout -- $FILE

echo "=== V4: use flash-layout unified attention instead of the gluon paged path"
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/attention_mha.py'); s = p.read_text()
old = "        self.use_flash_layout = False"
assert old in s, "V4 anchor missing"
p.write_text(s.replace(old, "        self.use_flash_layout = True"))
PY
judge v4_flashlayout
git checkout -- $FILE

echo "=== restore canonical server"
bash scripts/start_server.sh 0 step0final 2>&1 | tail -1
echo "=== DONE"
