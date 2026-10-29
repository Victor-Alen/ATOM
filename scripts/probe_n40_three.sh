#!/bin/bash
# Three n=40 determinism rounds. Verdict rule (from RESULTS.md):
#   C-0003 modal_rate must clear 0.225 + 2*0.066 ~= 0.36, C-0000 must stay 1.000.
# Baseline: short64 0.600 | C-0000 1.000 | C-0003 0.225
set -u
cd /workspace/ATOM
MOE=atom/model_ops/moe.py
ATTN=atom/model_ops/attention_mha.py

run_round () {
  local tag=$1
  echo "=== ${tag} ==="
  if ! bash scripts/start_server.sh 0 "n40_${tag}" >/dev/null 2>&1; then
    echo "  ${tag}: START FAILED"; return 1
  fi
  # never trust /health alone
  if ! curl -sf -m 90 -X POST http://127.0.0.1:8000/v1/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","prompt":"def add(a, b):\n    return","max_tokens":4,"temperature":0.0,"ignore_eos":true}' \
       | grep -q token_ids; then
    echo "  ${tag}: SERVER CANNOT GENERATE"; return 1
  fi
  python3 scripts/determinism_probe.py --tag "$tag" --repeat 40 --include-short \
    --out "/workspace/bench/det40_${tag}.json" 2>&1 | grep -E "modal_rate" | sed 's/^/  /'
}

patch_topk () {
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/moe.py'); s = p.read_text()
old = "from atom.model_ops.topK import rocm_aiter_topk_softmax as fused_topk"
assert old in s, "topk anchor missing"
new = '''from atom.model_ops.topK import rocm_aiter_topk_softmax as _aiter_topk_softmax


def fused_topk(
    gating_output, topk, renormalize, num_fused_shared_experts=0,
    num_routing_experts=0, fused_shared_experts_scoring_func=None,
):
    scores = torch.softmax(gating_output.to(torch.float32), dim=-1)
    weights, ids = torch.topk(scores, topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return weights, ids.to(torch.int32)'''
p.write_text(s.replace(old, new))
PY
}

# ---- R1: deterministic gating top-k
patch_topk
run_round r1_torchtopk
git checkout -- $MOE

# ---- R2: single context partition on every layer
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/attention_mha.py'); s = p.read_text()
old = """            context_partition_size = 256
            if self.sliding_window > 0:
                max_context_partition_num = 1
                context_partition_size = 128
"""
assert old in s, "onepart anchor missing"
p.write_text(s.replace(old, old + "            max_context_partition_num = 1\n"))
PY
run_round r2_onepart
git checkout -- $ATTN

# ---- R3: aiter fused MoE instead of the Triton MoE kernel
export ATOM_USE_TRITON_MOE=0
run_round r3_aitermoe
unset ATOM_USE_TRITON_MOE

echo "=== restore canonical server ==="
bash scripts/start_server.sh 0 clean0 2>&1 | tail -1
curl -sf -m 90 -X POST http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","prompt":"def add(a, b):\n    return","max_tokens":4,"temperature":0.0,"ignore_eos":true}' | grep -o '"token_ids":[^]]*' | head -1
echo "=== git ==="; git status --short | grep -v '^??' | head; echo "(only untracked scripts expected)"
echo "=== DONE"
