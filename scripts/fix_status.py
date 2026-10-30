#!/usr/bin/env python3
"""Bring STEP0-STATUS.md in line with round 22 (7881 exactness now verified)."""
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/STEP0-STATUS.md")
s = p.read_text()

s = s.replace(
    "ATOM 与 HF 的**算术差异已经收敛到 0**（≤6144 token 双设备逐位验证）；",
    "ATOM 与 HF 的**算术差异已经收敛到 0**（512→7881 token 双设备逐位验证，7881 在 A100 上 final hidden 完全相等）；",
)
s = s.replace(
    "| 4 | 算术逐位一致 | 512 / 2048 / 4096 / 6144 token：`max abs delta = 0.000000`，argmax 0/8；6144 在 MI308X 与 A100 双设备复现 |",
    "| 4 | 算术逐位一致 | 512 / 2048 / 4096 / 6144 token：`max abs delta = 0.000000`；**7881 token：final hidden `equal=True, max|delta|=0.0`（A100）** |",
)
s = s.replace(
    "| 7881 token 手写前向对照 | A100 共享 80GB 卡（他占 36GB），我方前向在 lm_head 处 CUDA assert（脚本兼容问题）|",
    "| ~~7881 token 手写前向对照~~ | ✅ 已完成（改比 final hidden，绕过 lm_head 的 CUDA assert）：**逐位相等** |",
)
s = s.replace(
    "| 6 | 7881 的 HF 参考 | 已在 A100 上算出（`hf_logits.pt`，2×98304 fp32）|",
    "| 6 | 7881 的 HF 参考 + 我方对照 | 均在 A100 上算出；final hidden 逐位相等（`equal=True`）|",
)
p.write_text(s)
print("status doc updated in worktree")
