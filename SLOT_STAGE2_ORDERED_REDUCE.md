# Mellum 生产 CK stage2：同配置逐 slot 输出与定序 reduce

2026-11-03 10:01 CST。实验用 `/workspace/ATOM-tritonmoe` 生产路径的 AITER/CK stage2、真实 L0 输入与 64 个 expert 权重；AITER 源码为 `8ddfc7510`。本次是**核级隔离探针**，没有将新路径设为生产服务默认，也没有做完整 token 验收。

## 实现

探针 `scripts/probe_moe_slot_stage2.py` 只做一次 sorting、一次 stage1，并冻结其 bf16 输出 `a2 [M,8,896]`。随后从同一 `a2` 和相同排序结果启动两次同一个 CK stage2：

1. 原路径：`topk=8`，各 expert 用 `AtomicAdd` 写入 `out [M,2304]`。
2. 逐 slot 路径：将排序 ID 中的 `(token, slot)` 映射成扁平行号 `token*8+slot`，把同一 `a2` 视为 `[M*8,1,896]`，以 `topk=1` 调用**同一个 stage2 函数**，输出 `per_slot [M,8,2304]`。每个有效 slot 恰好映射一次，已在运行时断言。每个输出行仅有一个 expert 贡献，不发生跨 expert 原子竞争。
3. 按 slot 0→7 将 `per_slot` 转为 fp32、逐次 `add_`，最后转换为 bf16。另保留转换前 fp32 结果用于比对。

两次 stage2 使用完全相同的 `w2`、sorted weights、sorted expert blocks、`num_valid_ids`、block_m、K/N、activation、dtype、stage2 函数与编译后 CK GEMM kernel 名；profiler 在两张卡的 M=1 和 M=866 实验均确认 kernel 名一致。为了改变地址映射，运行时 `topk` 参数从 8 变成 1，`a2` 的逻辑行数从 M×8 展平到 M×8，输出行数变为 M×8。**因此“同配置”指相同的 GEMM kernel/tile 和相同输入数值，不是完全相同的运行时参数。** CK stage2 launcher 的 dispatch 参数不含 token 数或 topk（`gemm_moe_ck2stages.cu:19-43,146-180`）；CK epilogue 从排序 ID 低 24 位取得写入行号（`gridwise_gemm_xdl_cshuffle_common.hpp:1793-1803`）。

## 结果：两卡各重复 40 次

两卡分别在独立进程中运行，`HIP_VISIBLE_DEVICES=0/1`。

| 输入 | GPU | 原 stage2 不同输出 | 逐 slot 不同输出 | 定序结果不同输出 | 原输出 vs HF max / mean | 定序 bf16 vs HF max / mean |
|---|---:|---:|---:|---:|---:|---:|
| decode M=1 | 0 | 40/40 | 1/40 | 1/40 | 0.0078125 / 5.958e-5 | 0.0078125 / 4.587e-5 |
| decode M=1 | 1 | 40/40 | 1/40 | 1/40 | 0.0004883 / 5.575e-5 | 0.0078125 / 4.587e-5 |
| prefill M=866 | 0 | 40/40 | 1/40 | 1/40 | 0.0625 / 2.186e-4 | 0.03125 / 1.6886e-4 |
| prefill M=866 | 1 | 40/40 | 1/40 | 1/40 | 0.0625 / 2.186e-4 | 0.03125 / 1.6886e-4 |

“不同输出”是 40 次中的不同 SHA-256 哈希数量，表中的原路径误差取该组第 1 次输出，因此会随原子累加调度波动。两张卡的**定序结果哈希逐位相同**：M=1 为 `583fec11fe6764c9`，M=866 为 `e0699e24c2f5dcb2`；原 stage2 两卡各 40 次的输出哈希集合无交集。M=866 的定序 bf16 相对 HF 的 mean 比本次原路径低约 22.8%，max 降一半。M=866 定序前的 fp32 结果相对 HF 为 `max 0.03076171875 / mean 1.9042e-4`。

本实验保持 stage1 与 CK stage2 GEMM 计算配置不变，并以每个 slot 只有一次写入消除 combine 竞争，因此直接支持：**本样本的跨 run/跨 GPU 非确定性来自 stage2 的共享 bf16 输出聚合；聚合方式也贡献了可测的平均误差。** 定序结果仍未与 HF 逐位相同，剩余误差可能包括 stage1/stage2 GEMM 算术、bf16 per-slot 写回及 HF 的不同累加路径。本实验不能推断原服务 L0 的 `max 0.14 / mean 8.2e-4` 全部由聚合造成，也没有证明 A/C token 达标。

原始证据：`/workspace/logs/step0_continuation/slot_stage2_gpu0.json`、`slot_stage2_gpu1.json` 及对应 `.log`。可重复脚本已入工作树；它会核对全部 M×8 个有效 slot 唯一映射，并通过 profiler 检查两次调用的 CK kernel 名一致。
