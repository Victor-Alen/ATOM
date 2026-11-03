from pathlib import Path

fixes = [
    ('REPORT-STEP0-2026-11-03.md',
     '- 2 × MI308X（gfx942），PCIe 互联，实测 P2P 49 GB/s',
     '- 2 × MI308X（gfx942），**xGMI 单链路互联**（1 hop / weight 15，见下方更正），实测 P2P 50.5 GB/s（1 GB 传输）'),
    ('TASK-STEP0-CONTINUATION.md',
     '- 硬件：2 × AMD MI308X（gfx942），每卡 192 GiB，80 CU；**PCIe 互联**，实测 P2P 带宽 49 GB/s',
     '- 硬件：2 × AMD MI308X（gfx942），每卡 192 GiB，80 CU；**xGMI 单链路互联**（amd-smi：XGMI / 1 hop / weight 15；RCCL topo dump 有 <xgmi count="1">），实测 P2P 带宽 50.5 GB/s（1 GB 传输），56 KiB 小传输 2.0 GB/s（延迟受限）'),
]
for name, old, new in fixes:
    p = Path('/workspace/ATOM-tritonmoe') / name
    s = p.read_text()
    if old in s:
        p.write_text(s.replace(old, new, 1))
        print('patched', name)
    else:
        print('MISS', name)