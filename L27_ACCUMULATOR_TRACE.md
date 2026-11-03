# L27 up projection：真实 CK 累加轨迹与单元素纠正

2026-11-03。已完成该单例在 CK kernel 内部的追踪和纠正实验。原 CK FP32 累加器最终值恰好为 35.375；独立块结果经 FP64 合并后，最终 FP32 值为 35.3749885559082，BF16 up 从 35.5 变为 35.25，gated 从 956 变为 948，与 HF 相同。这是一个选定元素的数值实验，不是已验收的全模型生产修复。

## 对象及真实执行路径

- 模型 L27，真实 A-0000 prefill 输入 M=990，固定输入/权重/路由。
- 输入位置 467，slot 3，expert 38，up feature 132，K=2304。
- CK stage1：block_m=128，V3、双 LDS，BF16 输入/权重、FP32 累加、BF16 输出。
- 定位后的 block_m_id=55、block_n_id=1、thread=4、C 寄存器标量偏移=41。
- 真实执行路径：`gridwise_moe_gemm.hpp::Run_2Lds` → `blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_v3.hpp::Run` → up 分支 `xdlops_gemm.Run`。

每次目标寄存器更新后写一个 FP32 快照，共 72 次，每次覆盖 32 个连续 K 元素。粒度是一次 XdlopsGemm 调用后的寄存器状态，不能看到该 MFMA 内部硬件的每个乘加节点。

## 从哪里开始出现差异

这里的归约累加器是 FP32，BF16 用于输入和最终输出。必须区分“FP32 与高精度前缀不同”和“输出 BF16 不同”：

| 已包含的 K 元素 | 原 CK FP32 累加器 | FP64 前缀点积 | CK 舍入 BF16 | HF 遮尾前缀实验 BF16 |
|---:|---:|---:|---:|---:|
| 32 | 0.7721967101 | 0.7721967250 | 0.7734375 | 0.7734375 |
| 2240 | 34.6109657288 | 34.6109559528 | 34.5 | 34.5 |
| 2272 | 35.1686210632 | 35.1686114959 | 35.25 | 35.25 |
| 2304 | **35.3750000000** | **35.3749875269** | **35.5** | **35.25** |

第一次记录到 FP32 与 FP64 前缀不同是在 K=32，误差 -1.4901161e-8。这是正常浮点舍入差异，并不单独证明 kernel 错误。前 71 个快照的 BF16 舍入结果都与 FP64 正确舍入及 HF 遮尾前缀实验一致；第 72 个、K=2304 时第一次出现 BF16 结果差异。

最终 CK 累加误差为 +1.2473087e-5，恰好把值推到 35.375。35.25 与 35.5 的 BF16 编码最低位分别为奇/偶，因此 round-to-nearest-even 对精确中点 35.375 选择 35.5。BF16 转换规则在这里执行正确；需要改善的是转换之前的 FP32 累加轨迹。

这些数据不能表述成“原 HF 累加器在 K=32 与 CK 首次分叉”。CK 数据来自真实内部寄存器；HF 对照是保持输入/权重张量形状，将目标权重行 K 后半段置零后重新运行 grouped_mm，观察输出 BF16。它不是原 HF kernel 的内部快照，也没有恢复原 HF 的 FP32 部分和。

## 内核中的纠正实验

对选定的这一个 up 元素，保留原 BF16 MFMA 计算，每次 XdlopsGemm 更新前将其 FP32 accumulator 清零，得到独立的 32 项块结果，再将这 72 个 FP32 块结果在 FP64 中合并。最后转回 FP32，进入既有 BF16 舍入和 gated epilogue。其余寄存器继续原计算路径。

该实验没有注入 HF 的 35.25，也没有以 FP64 完整点积替换 CK GEMM；修正值来自 kernel 自己实际计算的 72 个块结果。每块 FP32 结果与相应 FP64 32 项点积的最大差为 8.1956387e-8，支持快照与连续 K 前缀的对应关系。

| 量 | 原 CK | 选定元素分块 FP64 合并 | HF |
|---|---:|---:|---:|
| 转 BF16 前的 FP32 up | 35.375 | 35.3749885559082 | 原 HF FP32 中间值未抓取 |
| BF16 up | 35.5 | **35.25** | 35.25 |
| BF16 gated | 956 | **948** | 948 |
| 最终 FP32 up 对 FP64 点积的绝对误差 | 1.2473087e-5 | **1.0289950e-6** | 未测 |

控制检查：

- 调试二进制关闭 tracing 后，完整 up 和 gated 张量与原二进制逐值相同（各 7,096,320 个值）。
- 开启 tracing 后，完整 up 张量仍与原二进制逐值相同；三次完整轨迹相同。
- 开启目标元素纠正后，完整 up 张量只改变 1 个元素，完整 gated 张量也只改变 1 个元素，均为目标位置。

## 哪些代码需要动，能否推广

插桩/纠正在独立 CK 工作树 `/workspace/ck-mellum-l27-trace`，分支 `mellum/l27-accumulator-trace`，提交 `c2a2564bd`，基于 `d6e03ebd9`：

1. `gridwise_moe_gemm.hpp`：将调试 buffer 和目标 block/thread/register 参数传入 pipeline，并在 epilogue 将逻辑输出元素映射到物理线程/寄存器。
2. `blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_v3.hpp`：在 up XdlopsGemm 更新前后插入选定寄存器快照与可选的独立块/FP64 合并。
3. 对应 `gufusion_v1.hpp`：仅兼容新增的可选参数，实际本例使用 V3，V1 不执行该追踪。

这是直接修改 CK 归约精度的可执行证明，说明此单例可通过内核内部数值改动纠正。推广到全部寄存器需要设计高精度/补偿累加策略并测量 VGPR、occupancy、吞吐和全模型一致率。没有测量通用版本的性能。更接近高精度点积也不保证所有值与 HF 的 FP32 算法 bit exact；若目标是 HF bit exact，仍需对齐 HF 的实际归约算法，而非只提高累加精度。

当前调试纠正通过指定 block/thread/register 只作用于该单例，默认关闭。它没有作为默认生产修复合入，也没有进行新的端到端或 Step 0 验收。

## 产物与复现

- ATOM 分支：`dsh/mellum-l27-accumulator-trace`，从 `04bcac9a` 继续。
- 主脚本：`scripts/trace_l27_up_accumulator.py baseline` 先保存原二进制输出；随后在独立 JIT/CK 环境运行同脚本 `trace`。
- `scripts/render_l27_trace.py` 导出 72 行 CSV、曲线和丰富后的 JSON。
- 补丁：`patches/mellum_ck_l27_accumulator_trace.patch`，基于 CK `d6e03ebd9`，已验证可应用。
- 独立 JIT：`/workspace/ck-mellum-l27-trace-jit`；二进制 SHA256：`8cc2c8e42eb07d9bb478b30ae404b2a0f51b63cb741d50129d7b0cc5c5d610a8`。
- 原始服务器产物：`/workspace/logs/step0_continuation/l27_accumulator_trace/`。
- 本地：`L27_ACCUMULATOR_TRACE_RAW.json`、`L27_ACCUMULATOR_TRACE.csv`、`L27_ACCUMULATOR_ERROR.png/svg`。

GPU 调试 buffer 由脚本在同一进程内分配并传入 kernel。运行结束后无 GPU handle 遗留，两张卡显存占用均为 0%。原安装和生产 JIT 未修改，未推送上游。
