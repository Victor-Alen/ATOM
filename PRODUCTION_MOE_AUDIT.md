# Mellum 生产路径 MoE 源码与固定路由核查

2026-11-03 09:43 CST。ATOM 工作树 `b3dc6092`，AITER 源码 `8ddfc7510`，MI308X/gfx942，bf16、64 experts、top-8、hidden 2304、intermediate 896，无量化。本报告只涉及生产路径的 MoE。此前误改的 PyTorch eager 默认路径已由 `b3dc6092` 撤销；未推送。

## 结论

1. 生产默认路径从 `MellumSparseMoeBlock` 进入 `FusedMoE`，单独执行 fused top-k routing，再进入 AITER `fused_moe`。本配置的专家计算是 **Opus sorting → CK stage1 → CK stage2**，并非单个融合 kernel。
2. CK stage2 使用 `AtomicAdd` 将各 expert 的加权结果直接累加进 bf16 `moe_buf`；它不是先写 per-expert 结果、再由独立 gather/reduce kernel 合并。`moe_buf` 在 sorting 内清零。stage1 的写入方式是 `Set`。
3. 固定真实 L0 输入、权重、top-k ids/weights 后，decode 1-token 和 prefill 866-token 的 routing 各 40 次逐位相同；冻结 sorting 与 stage1 中间结果后，stage1 各 40 次逐位相同，**stage2 各 40 次产生 40 种输出**。因此已测这组输入的非确定性在 stage2，而非 top-k 并列选择。两张 GPU 以独立进程各跑 40 次，结论相同，且两卡输出哈希集合无交集。
4. 原 `ATOM_PROD_DUMP` 的 L0 `max 0.14 / mean 8.2e-4` 是历史测量，本次没有在相同服务请求上重现。本次捕获的 866-token L0 输入，融合 MoE 对 HF 为 `max 0.0625 / mean 2.186e-4`。八个单路 CK 调用再按固定顺序求和，对 HF 为 `max 0.03125 / mean 1.689e-4`，在这组输入上平均误差约降 22.7%。单路调用会改变 kernel 配置，不能据此断言原始 0.14 全由原子累加导致；stage1/CK GEMM 的系统性数值差异仍需单独量化。

## 源码链路

- `atom/models/mellum.py:102-110,154-194`：默认 `self.experts(...)`；只有显式 `ATOM_PROD_MOE=ref` 才走诊断参考分支。
- `atom/model_ops/moe.py:580-595,654-662,2955-2963,3007-3010`：`FusedMoE` 的 softmax top-k 与 AITER `fused_moe` 分开调用。
- `/app/aiter-test/aiter/fused_moe.py:336-394,423-450,1655-1661,1675-1691,1769-1787`：配置选择、sorting、`a2`、stage1/stage2 与输出。
- `/app/aiter-test/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh:214-227,282-298,315-325`：stage2 以 `IsInputGemm=false` 构造 `DeviceMoeGemm`，输出指针直接传给它。
- `/app/aiter-test/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp:280-281`：`IsInputGemm ? Set : AtomicAdd`，这是写入语义的直接证据。
- `/app/aiter-test/aiter/fused_moe.py:37-108` 与 `/app/aiter-test/csrc/include/moe_sorting_opus.h:425-443`：buffer 分配和 sorting 内清零。

## 实际调用与 buffer

对一次 `aiter.fused_moe` 调用用 `torch.profiler` 观察 GPU kernel；**下表不含此前独立的 routing 和 gate GEMM**。

| 输入 | 实测 kernel 启动 | sorting | block_m | sorting workspace |
|---|---:|---|---:|---:|
| decode, M=1 | 3 | 1 个 Opus | 32 | 0 B |
| prefill, M=866 | 4 | 2 个 Opus phase | 128 | 57,856 B |

两种输入均为 2stage、`ksplit=0`，其余各一次 CK stage1 和 CK stage2。实际 tensor 布局如下；`sorted_ids` 使用扁平的 token/slot 索引，`sorted_expert_ids` 按专家 block，`moe_buf` 为按 token 行存放的最终输出。

| buffer | M=1 | M=866 | dtype |
|---|---|---|---|
| `sorted_ids` | `[2048]` | `[15112]` | int32 |
| `sorted_weights` | `[2048]` | `[15112]` | fp32 |
| `sorted_expert_ids` | `[64]` | `[119]` | int32 |
| `num_valid_ids` | `[2]` | `[2]` | int32 |
| `a2` stage1 output | `[1,8,896]` | `[866,8,896]` | bf16 |
| `moe_buf` stage2 output | `[1,2304]` | `[866,2304]` | bf16 |

M=866 的 `a2` 为 12,414,976 B，`moe_buf` 为 3,990,528 B。没有独立的 per-expert 输出及 gather/reduce buffer。

## 隔离实验和界限

输入取 `/workspace/logs/hf_moe_l0.pt`，router ids/weights 与 `/workspace/logs/gate_engine.pt` 逐位核对。全部 64 个 L0 expert 权重从 checkpoint 装载并按生产方式 shuffle；expert 0 与既有 `/workspace/logs/prodw.pt` 逐位相同。探针：`scripts/probe_moe_fixed_router.py`。原始输出：`/workspace/logs/step0_continuation/fixed_router_stages.json` 与 `fixed_router_gpu1.json`。

| 探针，每种 40 次 | GPU0 | GPU1 |
|---|---:|---:|
| routing 输出哈希种类，M=1 / 866 | 1 / 1 | 1 / 1 |
| 固定 routing 的融合 MoE 输出哈希种类，M=1 / 866 | 40 / 40 | 40 / 40 |
| 冻结 sorting 后 stage1 输出哈希种类，M=1 / 866 | 1 / 1 | 1 / 1 |
| 冻结 stage1 后 stage2 输出哈希种类，M=1 / 866 | 40 / 40 | 40 / 40 |

两卡独立进程的输出哈希集合交集为 0。此前一次单进程切换双卡触发 ROCm memory access fault，该次结果已作废，不用于结论。

在 M=866 的固定输入上，stage2 重复输出最大差 `0.03125`；融合输出相对 HF 的 max/mean 为 `0.0625 / 2.186e-4`。分 slot 定序求和的数值仅作为对照，它同时改变了 top-k 形状与 kernel 选择，不能把精度改善直接当作 atomic 单因素因果效应。

## 下一步定位

若要拆清平均误差的来源，应在**相同 stage1 `a2` 与相同 stage2 GEMM 配置**下，让 stage2 先写 `[M,8,2304]` 的每 slot 结果，再做固定顺序、明确累加 dtype 的 reduce，与当前 bf16 `AtomicAdd` 逐位对比。这样才能将 expert GEMM 误差和 combine 误差拆开。改造位于 AITER stage2/输出写入处，不应再改 PyTorch 参考路径。之后需要在完整生产服务上复测 token 一致性、吞吐及 A/C 场景；本报告没有声称 Step 0 达标。
