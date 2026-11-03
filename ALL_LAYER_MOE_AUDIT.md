# CK GEMM、全层 SiLU/乘法与路由权重审计

2026-11-03。此前“CK 与 PyTorch 的 SiLU、乘法逐值一致”仅适用于 L0 的两个捕获，不应推广到全部层。本次补充真实生产请求 A-0000 的全部 28 层 prefill（990 个位置）和首个 decode（1 个位置），共 56 组；每组固定该层实际输入及路由，再比较 CK 与 HF 算子。该测试覆盖一个请求的两个阶段，不代表任意 prompt 或所有 decode 步。

## GEMM 差异定位

GPU profiler 显示 HF grouped_mm 执行 `Cijk_Alik_Bljk_...` 系列 kernel，具有 MT32x16x512、MT32x48x256、MT64x32x256 等不同 tile；生产 CK 执行 `ck::kernel_moe_gemm_2lds<...>`。两边不是同一个 GEMM 实现，分块及 MFMA 累加路径不同。BF16 输入相同不保证 FP32 累加结果逐位相同；最终结果位于 BF16 舍入边界附近时，微小累加偏差可以改变 BF16 输出。stage1 是无 split-K 的 Set 写出；本次不是 stage2 原子竞争问题。

一个实测例子来自 L27 prefill，输入位置 467、slot 3、expert 38、up 特征 132：

| 项目 | 数值 |
|---|---:|
| 同输入/权重的 FP64 点积 | 35.37498752691317 |
| 相邻 BF16 结果的分界点 | 35.375 |
| FP64 结果距离分界点 | -1.2473e-5 |
| HF GEMM up 输出 | 35.25 |
| CK GEMM up 输出 | 35.5 |
| 两边相同的 gate / SiLU 输出 | 26.875 |
| HF 最终 gated | 948 |
| CK 最终 gated | 956 |

此例 HF 舍入与 FP64 对照一致，CK 落在分界点另一侧。此前 L0 的 FP64 对照也有 CK 正确、HF 偏离的元素，不能据此宣布某个后端普遍更准确。当前定位到 GEMM 数值路径及输出舍入边界，尚未抓取两边逐个 K 分块的 FP32 accumulator，不能指出首次偏差发生在哪条 MFMA/哪次加法。

## 全部层的同输入隔离结果

所有 56 组重新计算的生产 CK 输出，都与服务捕获输出逐值相同，确认离线测试复现了实际生产路径。

| 对照 | 不同标量 / 总标量 |
|---|---:|
| GEMM gate：CK vs HF | 39,595 / 198,897,664 |
| GEMM up：CK vs HF | 45,820 / 198,897,664 |
| 相同 gate 输入下的 SiLU | 239 / 198,897,664 |
| 相同 SiLU/up 输入下的乘法 | 0 / 198,897,664 |
| 对齐舍入后的 stage1 gated vs HF gated | 75,345 / 198,897,664 |

SiLU 的差异仅出现在 L26 prefill（173 个）与 L27 prefill（66 个）。例如输入 -88.5：CK 为负零，PyTorch 为 -3.2473e-37；所有差异最大为 8.7574e-37。CK 公式为 `x * (1 / (1 + exp(-x)))`，PyTorch 为 `x / (1 + exp(-x))`。在此负值区间，中间倒数会进入 FP32 subnormal；当前 CK 编译启用 flush-denormals-to-zero，与观测到的负零相符。最终 PyTorch 输出本身仍可为正常浮点数。这一微小数值差异不能写成全层 bit exact，但不足以解释当前常规量级的 MoE 偏差。

PyTorch v2.10.0 公式来源：https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/ActivationSiluKernel.cu 。CK 来源：`include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp` 的 `Silu`。BF16 舍入对齐位于当前实验 CK 工作树 `gridwise_moe_gemm.hpp` 的 Mellum epilogue。

深层 gated 的绝对误差也不能直接套用 L0：本请求 L0 prefill aligned stage1 mean 为 1.910e-8，L13 为 5.662e-7，L27 为 1.203e-5，L27 最大差为 8（上面的具体例子）。这既受 GEMM 差异影响，也受该层数值幅度和 gating 乘法影响。

## 新发现：生产路由权重遗漏 HF 的 BF16 舍入

HF `MellumTopKRouter.forward` 在 FP32 softmax、top-k 与归一化之后，调用 `router_top_value.to(router_logits.dtype)`，即 BF16。ATOM `topK.py` 使用 FP32 top-k 权重，生产 helper 原先直接交给 CK；捕获的 221,984 个权重中，221,982 个在转 BF16 后数值改变。

此前 L0 探针注入的是 HF 捕获权重（恰好已舍入为 BF16），因此没有覆盖这处真实生产差异。之前 L0 的超低 mean 不等于生产各层在实际 FP32 权重下的误差。

固定相同的层输入/路由，stage2 down 舍入对齐在 56/56 组都降低 mean；再将实际路由权重舍入为 BF16，56/56 组又进一步降低 mean。对所有输出标量加权的 mean：stage1-only 0.0056611；stage1+stage2 0.0045940；再加 BF16 路由权重 3.4106e-5。这是固定输入的局部 MoE 对照，不能当成自由生成精度。

| prefill 层 | stage1+stage2、实际 FP32 权重的最终 MoE mean | 再对齐路由权重 BF16 的 mean |
|---|---:|---:|
| L0 | 9.055e-5 | 4.454e-7 |
| L13 | 1.646e-3 | 1.491e-5 |
| L26 | 1.396e-2 | 8.660e-5 |
| L27 | 3.743e-2 | 2.632e-4 |

加入显式实验开关 `ATOM_MELLUM_MOE_BF16_ROUTER_WEIGHTS=1`，在生产 helper 的排序之前执行 BF16 舍入，再以 FP32 存储以满足现有排序接口。这没有改变 top-k ids，没有用 HF GEMM 替换 CK。开关默认关闭；后续若合入默认，应考虑在 Mellum 路由/排序 kernel 中融合该舍入，避免两次独立 dtype 转换开销。此轮没有测量性能。

## 为什么局部误差更低而 token 一致率下降

这是两个不同的量：逐层固定输入的隐藏状态误差，以及 28 层自由生成的 token 一致率。实测 stage2 对齐在全部 56 组中降低局部 mean，不能称为 stage2 kernel 使局部误差变大。完整模型仍存在其他算子、路由权重、GEMM 数值路径的差异；argmax 是离散决策，一次 token 分叉后后续上下文和 KV 也会不同。局部平均误差下降不构成 token 一致率单调提高的保证。

此前将所有残差概括为舍入累积，归因不足。新增的路由权重遗漏是已确认的语义差异；对它的生成复测如下：

| 候选 | A20×64 | B20×64 | C20×8 |
|---|---:|---:|---:|
| stage1 舍入对齐 | 456/1280 = 0.3563 | 1280/1280 | 122/160 = 0.7625 |
| stage1+stage2 舍入对齐 | 284/1280 = 0.2219 | 1280/1280 | 121/160 = 0.7563 |
| 再对齐路由权重 BF16 | 323/1280 = 0.2523 | 1280/1280 | 141/160 = 0.8813 |

A/B 使用已核验 eager 参考，C 使用本次续作现场重算且与保存 eager 前缀 20/20 相同的参考。路由权重修正改善了该候选的 A/C，但 Step 0 仍未通过。首次导致 token 改变的 logits/argmax 及其完整逐层因果链尚未追踪，不能把 A 的下降完全归因于误差抵消或某一 kernel。

## 版本与复现

ATOM 分支 `dsh/mellum-all-layer-moe-audit`，从 `7fc94dd3` 继续。CK 仍使用既有实验版本 `d6e03ebd9`；这一轮没有改 GEMM 主循环或 SiLU 公式。helper 新增可选层名和采集开关，`UnquantizedFusedMoEMethod.apply` 传入层名，没有修改带 torch compile decorator 的模型文件。

采集必须显式设置 `ATOM_MELLUM_MOE_CAPTURE_DIR`，并在服务 ready 后创建目录中的 `ARMED` 文件，才会保存每层首个 prefill/decode。首轮调试曾捕获启动预热 M=16384；这些数据已排除。本报告仅使用 `all_layers_request_stage12_capture` 中 M=990/1 的真实请求，探针对此做形状断言。

复现：指定 CK_DIR=/workspace/ck-mellum-hf-stage12、AITER_JIT_DIR=/workspace/ck-mellum-hf-stage12-jit、CK_MELLUM_STAGE1_MODE=1、CK_MELLUM_STAGE2_ROUND_WEIGHT=1 后运行：

```bash
python3 scripts/probe_all_layer_moe.py \
  --captures /workspace/logs/step0_continuation/all_layers_request_stage12_capture \
  --output /workspace/logs/step0_continuation/all_layer_request_moe_probe.json
```

原始结果 `ALL_LAYER_MOE_PROBE_RAW.json`；逐层表 `ALL_LAYER_MOE_METRICS.csv`；FP64 边界例子 `GEMM_BOUNDARY_EXAMPLE.json`。服务及探针均已退出，GPU handle owner 为 0，两张 GPU 显存回到 0%。未推送上游。默认配置未改变，本轮没有进行 C200、MTP、并发/调度验证。
