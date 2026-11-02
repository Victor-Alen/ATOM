from pathlib import Path

p = Path('/workspace/ATOM-tritonmoe/BRIEF-0800.md')
s = p.read_text()
i = s.find('### 2026-11-03 07:50 (round 96)')
if i > 0:
    s = s[:i].rstrip() + chr(10)
anchor = '## 建议的下一步（按优先级）'
block = (
    '## 目标指定的方案：已实现，已测试，无效' + chr(10) + chr(10) +
    '目标要求「生产路径接入 eager 同序 attention + HF MoE dispatch」。两个都实现了，同时启用测了：' + chr(10) + chr(10) +
    '```' + chr(10) +
    '生产基线                          :  42/192 = 0.2188' + chr(10) +
    '只换 eager attention              :  13/192 = 0.0677' + chr(10) +
    '只换 HF MoE                       :  39/192 = 0.2031' + chr(10) +
    '两者同时（目标指定配置）          :  12/192 = 0.0625   ← 最低' + chr(10) +
    '```' + chr(10) + chr(10) +
    '**指定组合比基线低、比任何单项也低。**' + chr(10) + chr(10) +
    '原因：attention 本来就准（4.9e-4），换掉它几乎不减少误差，只改变舍入路径；MoE 替换' + chr(10) +
    '同样改变它参与的所有位置的舍入。两次替换把轨迹推向不同方向，换个方向不等于靠近 HF。' + chr(10) + chr(10) +
    '**唯一能达到 HF 精确一致的配置仍是完整参考路径**（端到端有序 PyTorch 运算），慢 3 倍。' + chr(10) + chr(10)
)
s = s.replace(anchor, block + anchor, 1)
p.write_text(s)
print('brief updated; lines:', len(s.splitlines()))