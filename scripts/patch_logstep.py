#!/usr/bin/env python3
"""Log the per-call logits summary (argmax, top-5, margin) so a step-by-step diff
against HF is possible without the flaky in-process probe."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
old = """            _logits = _F.linear(_hs, self.lm_head.weight)"""
new = """            _logits = _F.linear(_hs, self.lm_head.weight)
            if os.environ.get("ATOM_REFERENCE_LOGSTEP"):
                try:
                    _row = _logits[-1].float()
                    _v, _i = torch.topk(_row, 5)
                    _mi = int(_i[0].item())
                    _margin = float(_v[0].item() - _v[1].item()) if _v.numel() > 1 else 0.0
                    _path = os.environ["ATOM_REFERENCE_LOGSTEP"]
                    with open(_path, "a", encoding="utf-8") as _fh:
                        _fh.write(
                            f"row={_logits.shape[0]} argmax={_mi} "
                            f"top5={[int(x) for x in _i.tolist()]} "
                            f"vals={[round(float(x), 4) for x in _v.tolist()]} "
                            f"margin={_margin:.4f}\\n"
                        )
                except Exception:  # noqa: BLE001
                    pass"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
ast.parse(s)
print("logits step logging added")
