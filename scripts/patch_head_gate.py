#!/usr/bin/env python3
"""Only use the all-rows lm_head when the reference path actually produced the
hidden states (keeps warmup and non-reference batches on ATOM's own head)."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()

old = """            _ref = reference_forward(self, input_ids, positions)
            if _ref is not None:
                return _ref"""
new = """            _ref = reference_forward(self, input_ids, positions)
            self._reference_used = _ref is not None
            if _ref is not None:
                return _ref"""
assert old in s
s = s.replace(old, new, 1)

old2 = """        if __import__("os").environ.get("ATOM_REFERENCE_MODE") == "1":
            import torch.nn.functional as _F

            return _F.linear(hidden_states, self.lm_head.weight)"""
new2 = """        if __import__("os").environ.get("ATOM_REFERENCE_MODE") == "1" and getattr(
            self, "_reference_used", False
        ):
            import torch.nn.functional as _F

            return _F.linear(hidden_states, self.lm_head.weight)"""
assert old2 in s
s = s.replace(old2, new2, 1)

# reset the flag at the start of every forward
old3 = """        if __import__("os").environ.get("ATOM_REFERENCE_MODE") == "1" and inputs_embeds is None:"""
new3 = """        self._reference_used = False
        if __import__("os").environ.get("ATOM_REFERENCE_MODE") == "1" and inputs_embeds is None:"""
assert old3 in s
s = s.replace(old3, new3, 1)
p.write_text(s)
ast.parse(s)
print("lm_head gated on reference usage")
