#!/usr/bin/env python3
"""Re-apply the greedy argmax patch and add env-gated tracing to ATOM's sampler."""
import subprocess, ast
from pathlib import Path

REPO = Path("/workspace/ATOM")
F = REPO / "atom/model_ops/sampler.py"
subprocess.run(["git", "checkout", "--", str(F)], cwd=REPO, check=True)
src = F.read_text()
NL = chr(10)

if "import json" not in src:
    src = src.replace("import warnings" + NL + NL + "import torch",
                      "import json" + NL + "import os" + NL + "import warnings" + NL + NL + "import torch", 1)

anchor = "        # No Top-K Top-P parameters, perform temperature-based sampling"
assert anchor in src

patch_lines = [
    "        # ---- temporary instrumentation (env-gated, remove before shipping) ----",
    "        _trace = os.environ.get('ATOM_SAMPLER_TRACE')",
    "        if _trace:",
    "            with torch.no_grad():",
    "                _t2 = logits.detach().float().topk(2, dim=-1)",
    "                _gaps = (_t2.values[:, 0] - _t2.values[:, 1]).tolist()",
    "                _am = logits.argmax(dim=-1).tolist()",
    "                _tv = temperatures.detach().float().flatten().tolist()",
    "            with open(_trace, 'a') as _fh:",
    "                _fh.write(json.dumps({'rows': int(logits.shape[0]),",
    "                                      'all_greedy': bool(all_greedy),",
    "                                      'filtering': self._needs_filtering(top_ks, top_ps),",
    "                                      'temps': _tv[:8],",
    "                                      'argmax': _am[:8],",
    "                                      'gap': [round(g, 6) for g in _gaps[:8]]}))",
    "                _fh.write(chr(10))",
    "",
    "        # Temperature 0 must be a true argmax: the plain-temperature path below",
    "        # runs a Gumbel / exponential race with FRESH noise on every call, so any",
    "        # exact logit tie at the maximum (bf16 resolution near |x|=10 is ~1e-2)",
    "        # becomes a coin flip. Honour all_greedy here, not only in the",
    "        # top-k / top-p branch, so greedy decoding is bit-reproducible.",
    "        if all_greedy:",
    "            return logits.argmax(dim=-1).to(torch.int)",
    "",
]
src = src.replace(anchor, NL.join(patch_lines) + anchor, 1)
F.write_text(src)
ast.parse(src)
print("patched + syntax ok")
