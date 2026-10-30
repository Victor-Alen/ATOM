#!/usr/bin/env python3
"""Debug compute_logits under reference mode."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = """            import torch.nn.functional as _F

            return _F.linear(hidden_states, self.lm_head.weight)"""
new = """            import torch.nn.functional as _F

            _logits = _F.linear(hidden_states, self.lm_head.weight)
            if __import__("os").environ.get("ATOM_REFERENCE_DEBUG") in ("1", "2"):
                _top = _logits.float().topk(2, dim=-1)
                print(
                    f"[ref-logits] shape={tuple(_logits.shape)} dtype={_logits.dtype} "
                    f"top1={_top.indices[0].tolist()} top1v={_top.values[0].tolist()} "
                    f"absmax={float(_logits.float().abs().max()):.4f}",
                    flush=True,
                )
            return _logits"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("logits debug added")
