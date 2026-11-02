from pathlib import Path

p = Path('/workspace/ATOM-tritonmoe/BRIEF-0800.md')
s = p.read_text()
i = s.find('### 2026-11-03 07:50 (round 95)')
if i > 0:
    s = s[:i].rstrip() + chr(10)
anchor = '## 建议的下一步（按优先级）'
block = (
    '## 三端口同批复测（目标要求的协议，已执行）' + chr(10) + chr(10) +
    '三台服务**同时**运行（8001 参考 + 8002 生产基线在 GPU0，8003 生产+有序MoE 在 GPU1），' + chr(10) +
    '同一批三条 A prompt、64 greedy token：' + chr(10) + chr(10) +
    '```' + chr(10) +
    '端口 8001  参考路径       : 192/192 = 1.0000   exact 3/3' + chr(10) +
    '端口 8002  生产基线       :  42/192 = 0.2188   exact 0/3' + chr(10) +
    '端口 8003  生产+有序MoE   :  39/192 = 0.2031   exact 0/3' + chr(10) +
    '```' + chr(10) + chr(10) +
    '参考路径在两台生产服务抢占同一张卡时仍是 192/192——确定性在资源竞争下成立（运算有序）。' + chr(10) + chr(10)
)
s = s.replace(anchor, block + anchor, 1)
p.write_text(s)
print('brief cleaned; lines:', len(s.splitlines()))