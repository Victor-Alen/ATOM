#!/bin/bash
# Determinism hunt round 2: is it the MoE gating top-k?
# Replace the aiter top-k gating kernel with a torch reference (torch.topk on a
# softmax) and re-run the repeat test.
set -u
cd /workspace/ATOM
MOE=atom/model_ops/moe.py

judge () {
  local tag=$1
  if ! bash scripts/start_server.sh 0 "det_${tag}" >/dev/null 2>&1; then
    echo "  ${tag}: START FAILED"; return 1
  fi
  # do not trust /health: require a real completion
  if ! curl -sf -m 60 -X POST http://127.0.0.1:8000/v1/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","prompt":"def add(a, b):\n    return","max_tokens":4,"temperature":0.0,"ignore_eos":true}' \
       | grep -q token_ids; then
    echo "  ${tag}: SERVER CANNOT GENERATE"; return 1
  fi
  python3 scripts/repeat_tokens.py --tag "$tag" --repeat 8 --include-short \
    --out "/workspace/bench/det_${tag}.json" 2>&1 | grep -E "distinct" | sed 's/^/  /'
}

echo "=== G1: deterministic torch top-k instead of the aiter gating kernel"
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/moe.py'); s = p.read_text()
old = "from atom.model_ops.topK import rocm_aiter_topk_softmax as fused_topk"
assert old in s, "G1 anchor missing"
new = '''from atom.model_ops.topK import rocm_aiter_topk_softmax as _aiter_topk_softmax


def fused_topk(
    gating_output,
    topk,
    renormalize,
    num_fused_shared_experts=0,
    num_routing_experts=0,
    fused_shared_experts_scoring_func=None,
):
    """Deterministic torch reference for the aiter top-k gating kernel."""
    scores = torch.softmax(gating_output.to(torch.float32), dim=-1)
    weights, ids = torch.topk(scores, topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return weights, ids.to(torch.int32)'''
p.write_text(s.replace(old, new))
print("patched moe.py gating")
PY
judge g1_torchtopk
git checkout -- $MOE

echo "=== G2: same, but also disable aiter GEMM autotuning lookup"
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/moe.py'); s = p.read_text()
old = "from atom.model_ops.topK import rocm_aiter_topk_softmax as fused_topk"
new = '''from atom.model_ops.topK import rocm_aiter_topk_softmax as _aiter_topk_softmax


def fused_topk(
    gating_output,
    topk,
    renormalize,
    num_fused_shared_experts=0,
    num_routing_experts=0,
    fused_shared_experts_scoring_func=None,
):
    """Deterministic torch reference for the aiter top-k gating kernel."""
    scores = torch.softmax(gating_output.to(torch.float32), dim=-1)
    weights, ids = torch.topk(scores, topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return weights, ids.to(torch.int32)'''
p.write_text(s.replace(old, new))
PY
export AITER_ENABLE_AUTOTUNE=0
export AITER_TUNE_MODE=none
judge g2_torchtopk_notune
git checkout -- $MOE

unset AITER_ENABLE_AUTOTUNE AITER_TUNE_MODE
echo "=== restore canonical server"
bash scripts/start_server.sh 0 clean0 2>&1 | tail -1
curl -sf -m 60 -X POST http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","prompt":"def add(a, b):\n    return","max_tokens":4,"temperature":0.0,"ignore_eos":true}' | head -c 160
echo
echo "=== git ==="; git status --short | head
echo "=== DONE"
