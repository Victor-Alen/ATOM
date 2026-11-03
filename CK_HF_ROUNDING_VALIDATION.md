# CK 中间舍入对齐与生产生成复测

日期：2026-11-03。结论：已在实际生产 CK 两阶段 kernel 内实现 HF 对应的 BF16 舍入边界，并完成独立环节和生产服务复测。stage1 的主要数值差异大幅下降，但 A/C 生成一致率仍未达到 0.90，Step 0 未通过。实验版本保持显式启用，现有默认生产路径仍为原 CK 算术加逐 slot 定序聚合。

## 修改了哪些 kernel

实际调用链：ATOM `UnquantizedFusedMoEMethod.apply` → `mellum_ordered_moe` → AITER `get_2stage_cfgs` / `ck_moe_stage1_fwd` / `ck_moe_stage2_fwd` → `DeviceMoeGemm` → `kernel_moe_gemm`。BF16 模块为 `module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so`。

| 位置 | 已实现改动 | 没有改动的部分 |
|---|---|---|
| CK stage1，`gridwise_moe_gemm.hpp` 的 SiLU epilogue | gate/up GEMM 累加值各转 BF16 再转 float；SiLU 输出再转 BF16；随后乘 up，最终按原路径写 BF16 | GEMM tile、MFMA 主循环、权重布局、SiLU 公式、排序和路由 |
| CK stage2，同一头文件的 down epilogue | 在乘路由权重之前，先将 down 累加值舍入为 BF16，再转回 float 乘权重，最终写 BF16 | down GEMM 主循环及既有逐 slot 输出寻址 |

正常 `Run` 和双 LDS `Run_2Lds` 都已覆盖；每种模式的带 scale / 不带 scale 实现同步修改。stage1 两个实测配置：M=1 使用 block_m=32、V1；M=866 使用 block_m=128、V3。每个模式继续使用原选定 GEMM 配置。没有用 HF/PyTorch GEMM 替换生产 stage1；新增舍入在 CK kernel 内执行，没有为这些舍入新增独立 kernel 或中间 buffer。

stage1 mode 0 保留原行为；mode 1 对齐舍入；mode 2/3/4 仅供隔离诊断，分别将 GEMM gate、GEMM up、舍入后的 SiLU 写入原 stage1 输出 buffer。诊断模式只在测试进程中启用，不用于生成。

## 分别验证 GEMM、SiLU、乘法

使用同一真实 L0 捕获，固定输入、权重、top-k ids/weights。HF grouped_mm 重建结果与捕获的 HF MoE 输出逐值相同。所有 mean/max 均为逐元素绝对误差。

| 对照 | M=866：不同值 / 总值 | mean | max |
|---|---:|---:|---:|
| CK GEMM gate BF16 vs HF gate | 1178 / 6,207,488 | 5.314e-8 | 0.0078125 |
| CK GEMM up BF16 vs HF up | 1149 / 6,207,488 | 4.950e-8 | 0.0078125 |
| 相同 CK gate 输入：CK SiLU vs PyTorch SiLU，输出 BF16 | 0 / 6,207,488 | 0 | 0 |
| 相同 CK SiLU/up 输入：CK 乘法 vs PyTorch 乘法，输出 BF16 | 0 / 6,207,488 | 0 | 0 |
| 原 stage1 gated vs HF | 2,574,913 / 6,207,488 | 1.118e-4 | 0.0625 |
| 对齐舍入后的 stage1 gated vs HF | 2052 / 6,207,488 | 1.495e-8 | 0.0078125 |

M=1：GEMM gate/up 各有 1/7168 个值与 HF 不同；同输入 SiLU、乘法均完全相同；stage1 gated 从 2974/7168 个值不同降到 2/7168，mean 从 2.935e-5 降到 6.813e-8。以上证明，在这两份输入上，不需要更换 SiLU 公式或逐元素乘法；主要改动是舍入边界。不能将该结论推广成任意输入均 bit exact。

进一步对 M=866 的 CK/HF 不同 GEMM 元素计算 FP64 点积，再直接按 BF16 nearest-even 规则舍入，避免先转 FP32 的二次舍入：gate 的 1178 个不同元素中，CK 有 536 个吻合该结果、HF 有 584 个吻合、58 个双方均不吻合；up 对应为 537、563、49。CK 与 HF 的 GEMM 累加均可能偏离这个 FP64 对照，不能仅凭 HF BF16 输出判 CK 算错。此检查只覆盖两边不同的元素。

## L0 MoE 最终输出

| 实际 CK 路径 | 对 HF mean | max | 不同标量 / 1,995,264 |
|---|---:|---:|---:|
| 原 CK stage1 + 原 stage2 权重融合 + 定序聚合 | 1.689e-4 | 0.03125 | 1,148,323 |
| 只对齐 stage1 舍入 | 9.137e-5 | 0.03125 | 743,671 |
| stage1 与 stage2 舍入都对齐 | 3.395e-7 | 0.015625 | 4585 |

这里是一次 L0 prefill 的 `[866,2304]` 隐藏状态张量，不是请求数或生成 token 数。此前的 164 个不同标量使用的是直接注入的 HF gated 中间结果；本次 4585 个则来自真正 CK stage1 的输出。两者不能混用。

## 生产服务端到端复测

同一端口 8003 顺序运行各候选，单客户端、BF16、level 0、eager、无 prefix cache。A/B 使用已核验的保存 HF eager 参考；C20 本次现场用 HF eager、BF16、greedy、use_cache=True 重算 8 token。新 C20 与原 eager 文件的前 8 token 在 20/20 条上相同。

| 生产候选 | A20 × 64 | B20 × 64 | C20 × 8，新 HF 参考 |
|---|---:|---:|---:|
| 定序 CK 基线 | 246/1280 = 0.1922 | 1280/1280 = 1.0000 | 130/160 = 0.8125 |
| 只对齐 stage1 | 456/1280 = 0.3563 | 1280/1280 = 1.0000 | 122/160 = 0.7625 |
| stage1 + stage2 对齐 | 284/1280 = 0.2219 | 1280/1280 = 1.0000 | 121/160 = 0.7563 |

基线 A/B 沿用之前同一生产 helper 的结果；基线 C 与两个候选的 C 均在本次用同一份新 HF 参考复测。这不是三端口同时同批验收。局部输出更接近 HF 并未令生成一致率单调改善，不能由 L0 的低 mean 推出 Step 0 通过。其他层、算子与后续路由/生成轨迹需要进一步定位；本次没有证据指定另外某个 kernel 为剩余误差根因。

确定性回归：关闭新舍入后，M=1/866 各 40 次的哈希与原实现完全相同；只对齐 stage1 时，各 40 次均唯一；两处对齐时，两张 GPU 各 40 次均唯一且跨 GPU 哈希相同。最终生产服务 `det3.py` 三次 16-token 输出相同。服务和测试进程均已退出，两张物理 GPU 显存占用回到 0%。

## 版本、复现和限制

- ATOM 分支：`dsh/mellum-ck-hf-rounding`，从 `f0ab45dc` 继续，保留原拓扑更正文档。首个诊断提交 `f55aa5e7`。最终提交见 git log。
- AITER 来源：`/app/aiter-test`，`8ddfc7510536d61f4e1863e9cf91c495be1f8c2c`；原安装及其已有 JIT 库未修改。
- CK 基线：`af7118e342580ecd3f71edce7b1d0ba465012ecf`。
- stage1 CK 工作树 `/workspace/ck-mellum-hf-round`，分支 `mellum/hf-stage1-rounding`，提交 `78a912528`。
- stage1+stage2 CK 工作树 `/workspace/ck-mellum-hf-stage12`，分支 `mellum/hf-stage12-rounding`，提交 `d6e03ebd9`。
- 完整补丁：`patches/mellum_ck_hf_rounding.patch`；已在未改动的 CK 基线上执行 `git apply --check`。
- 独立 JIT 目录分别为 `/workspace/ck-mellum-hf-jit`、`/workspace/ck-mellum-hf-stage12-jit`。二进制与参考文件 SHA256 在 `ck_hf_rounding_summary.json`。

服务显式启用命令（启动前先确认没有旧服务）：

```bash
cd /workspace/ATOM-tritonmoe
HIP_VISIBLE_DEVICES=0 scripts/start_mellum_ck_hf_round.sh stage1
# 或 stage12；日志会打印 CK 提交与二进制 SHA256
```

复现实验：`scripts/probe_ck_stage1_rounding.py <output.json> [--decode]`，并配套上述 CK_DIR、AITER_JIT_DIR、CK_MELLUM_STAGE1_MODE=1；stage12 还需 CK_MELLUM_STAGE2_ROUND_WEIGHT=1。脚本分别隔离 GEMM、SiLU、乘法、最终 MoE；`scripts/verify_ck_round_repeat.py` 做 40 次重复；`scripts/summarize_ck_rounding.py` 汇总已保存结果。

原始日志和 JSON：服务器 `/workspace/logs/step0_continuation/ck_*round*`。本地 `CK_HF_ROUNDING_RAW.json` 汇总全部探针和验收数字。未测新版本吞吐、C200 全量、MTP、并发或调度压力；没有推送上游。该候选未达到精度门槛，因此没有改成默认生产配置。
