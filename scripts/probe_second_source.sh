#!/bin/bash
# Round 2b: gating top-k was confirmed as one source (short64: 3 -> 1 distinct).
# C-0003 still varies at 6183 tokens, so there is a second source at long
# context. Combine the deterministic gating with the single-partition patch to
# see whether the second source lives in the partition path after all.
set -u
cd /workspace/ATOM

judge () {
  local tag=$1
  if ! bash scripts/start_server.sh 0 "det_${tag}" >/dev/null 2>&1; then
    echo "  ${tag}: START FAILED"; return 1
  fi
  if ! curl -sf -m 60 -X POST http://127.0.0.1:8000/v1/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","prompt":"def add(a, b):\n    return","max_tokens":4,"temperature":0.0,"ignore_eos":true}' \
       | grep -q token_ids; then
    echo "  ${tag}: SERVER CANNOT GENERATE"; return 1
  fi
  python3 scripts/repeat_tokens.py --tag "$tag" --repeat 8 --include-short \
    --out "/workspace/bench/det_${tag}.json" 2>&1 | grep -E "distinct" | sed 's/^/  /'
}

patch_topk () {
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/moe.py'); s = p.read_text()
old = "from atom.model_ops.topK import rocm_aiter_topk_softmax as fused_topk"
assert old in s
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

patch_onepart () {
python3 - <<'PY'
from pathlib import Path
p = Path('atom/model_ops/attention_mha.py'); s = p.read_text()
old = """            context_partition_size = 256
            if self.sliding_window > 0:
                max_context_partition_num = 1
                context_partition_size = 128
"""
assert old in s
p.write_text(s.replace(old, old + "            max_context_partition_num = 1\n"))
PY
}

echo "=== H1: deterministic gating + single context partition"
patch_topk
patch_onepart
judge h1_topk_onepart
git checkout -- atom/model_ops/moe.py atom/model_ops/attention_mha.py

echo "=== H2: deterministic gating + single partition + no MoE kernel change (baseline assert: short64=1)"
patch_topk
judge h2_topk_only
git checkout -- atom/model_ops/moe.py

echo "=== restore canonical server"
bash scripts/start_server.sh 0 clean0 2>&1 | tail -1
curl -sf -m 60 -X POST http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/Mellum2.1-12B-A2.5B-Thinking","prompt":"def add(a, b):\n    return","max_tokens":4,"temperature":0.0,"ignore_eos":true}' | head -c 120
echo
echo "=== git ==="; git status --short | head
echo "=== DONE"
