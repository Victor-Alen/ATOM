# CK V3 up 分块高精度合并修正与 L27 验收

2026-11-03。指定真实 L27 位置的 BF16 up 验收通过：HF 与新版 CK 都为 **35.25，BF16 编码 0x420D**。gate×SiLU 后的 gated 结果两边都为 948。实现已去掉先前调试版的 token/expert/thread/register 选择逻辑，统一作用于 BF16 V3 up 的全部输出寄存器。

## 归约方式如何修改

原方式：BF16 MFMA 每次把新乘积继续累加到同一个 FP32 accumulator；K 越长，已有舍入偏差会参与后续计算。

修正方式：每次 up `xdlops_gemm.Run` 前将对应 FP32 accumulator 清零，独立算出当前块的部分和，再将此部分和转 FP64 并累加到独立的 FP64 状态。合并结果转回 FP32 后进入原来的输出 epilogue。本例共 72 块、每块 32 项：

```text
partial32 = BF16_MFMA(x_chunk, w_chunk, initial_accumulator=0)
sum64 += FP64(partial32)
up32 = FP32(sum64)
up_bf16 = BF16_round(up32)
```

gate 的原归约过程保持不变。全部变化位于 `blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_v3.hpp`，35 行新增代码；没有新增显式中间 tensor 或独立 GPU kernel，没有调用 HF/PyTorch GEMM，也没有注入参考输出。

此实现适用于 BF16 V3 gufusion 路径；V1 未修改。该实例是 M=990、block_m=128 的 prefill V3，不代表 decode V1 也被改成高精度归约。

## 最终指定位置验收

固定服务器生产捕获的 L27 输入、权重和路由：输入位置 467、slot 3、expert 38、up feature 132、K=2304。

| 项目 | 原 CK | 新 CK，HIP_VISIBLE_DEVICES=0 | 新 CK，HIP_VISIBLE_DEVICES=1 | HF |
|---|---:|---:|---:|---:|
| BF16 up | 35.5 | **35.25** | **35.25** | **35.25** |
| BF16 up 编码 | 0x420E | **0x420D** | **0x420D** | **0x420D** |
| BF16 gated | 956 | **948** | **948** | **948** |

两张卡分别重复 40 次，每次完整 up 张量哈希均唯一且跨卡相同：`2f4fb75dc93d1c5dcd242e76188eefc3142f2434e6ed48a5ae0539384519c561`。完整 gate 张量与原 CK 基线数值相同（7,096,320/7,096,320）。

本轮通用版本不含调试快照代码，验收直接比较最终 BF16 up/gated；此前的单元素轨迹已经证明分块 FP64 合并能将舍入前结果留在 35.375 分界点下方。

## 全张量统计与性能

验收指定位置已一致，整张 L27 up 尚未与 HF bit exact：

| 全张量对 HF | 原 CK | 新 CK |
|---|---:|---:|
| up 不同标量 / 7,096,320 | 1324 | **1154** |
| up max 绝对误差 | 0.25 | **0.125** |
| up mean 绝对误差 | 1.863169e-6 | **1.688642e-6** |
| gated 不同标量 / 7,096,320 | 2191 | **2046** |
| gated max 绝对误差 | 8 | **2** |
| gated mean 绝对误差 | 1.203499e-5 | **1.108158e-5** |

高精度归约并不复刻 HF 的实际 FP32 归约顺序，不能保证全部输出都与 HF 逐位相同。本轮目标是纠正指定案例，并报告全层影响。

同一 GPU、真实 M=990 数据、固定排序与权重，10 次 warmup、50 次同步 wall-clock 样本：原 stage1 调用中位数 **0.782714 ms**，新版本 **1.606979 ms**，延迟为 **2.053×**。该测量包括 stage1 的 Python 调度与同步；不包括完整 MoE、排序或服务生成，不能换算成服务 token/s。原基线在新版本编译完成后重新测量，避免编译 CPU 负载影响对照。

额外 FP64 状态有明确性能代价，因此新版本保持独立显式启用。全模型吞吐及 A/B/C 没有在该通用版本上重新验收。

## 生产入口与版本管理

- CK 基线 `d6e03ebd9`；修正分支 `mellum/up-block-fp64`，提交 **`14e5a4558`**。
- CK 工作树 `/workspace/ck-mellum-up-fp64`，独立 JIT `/workspace/ck-mellum-up-fp64-jit`。
- ATOM 分支 `dsh/mellum-up-block-fp64`，从 `277aea37` 继续；最终提交见 git log。
- 补丁 `patches/mellum_up_block_fp64.patch`，已在未改动的 CK `d6e03ebd9` 上执行 `git apply --check`。
- JIT 二进制 SHA256：`e7e648c5b818975b7d4e1eefd10b0b30e0c86bcce1db06d27c3df073c61b18e9`。

显式启动生产候选（先确认没有旧服务）：

```bash
cd /workspace/ATOM-tritonmoe
HIP_VISIBLE_DEVICES=0 scripts/start_mellum_up_fp64.sh
```

入口固定 CK/JIT 路径并打印源码提交及二进制 SHA256；启用既有 stage1 HF 舍入、stage2 down 舍入和逐 slot 定序聚合。路由权重 BF16 开关仍按现有配置控制，本次没有修改它的默认值。

已经通过真实服务冒烟：8003 服务加载该二进制、GPU 显存实际占用 42%，A-0000 单请求成功完成 prefill 和 decode，2/2 token 与 HF 前缀相同。此结果只证明入口可运行，不是 Step 0 验收。

验证脚本 `scripts/validate_up_fp64.py` 在同输入/权重上重新计算 HF grouped_mm，并同时检查指定位置、完整张量误差、40 次重复和 stage1 延迟。`scripts/collect_up_fp64_validation.py` 汇总双卡、基线与服务冒烟数据并检查验收断言。

完整原始汇总：`UP_FP64_FIX_VALIDATION_RAW.json`；服务器 `/workspace/logs/step0_continuation/up_fp64_*`。原安装、原 JIT 与默认启动路径未覆盖，未推送。测试结束后服务已正常退出，GPU handle owner=0，两张 GPU 显存均为 0%。
