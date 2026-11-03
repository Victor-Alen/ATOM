# Mellum 生产路径：CK stage2 逐 slot 定序聚合

2026-11-03。服务器本地仓库：`/workspace/ATOM-tritonmoe`，分支 `dsh/mellum-ordered-stage2`。基线 `07c11d40`；实现提交 `f0aed701`。未推送上游。

## 修改与范围

生产 `FusedMoE` 在 `config.model_type == "mellum"`、未量化 BF16、TP=DP=EP=1 时默认调用 `atom/model_ops/mellum_ordered_moe.py`。保留 AITER 排序和原 CK 两阶段 GEMM：stage1 输出 `[M,8,896]`；将排序后的 `(token,slot)` 映射到唯一输出行；stage2 用相同 CK GEMM 写 `[M×8,2304]`；按 slot 0→7 顺序以 FP32 累加，再转 BF16。运行时 `topk` 从 8 变 1，仅用于 stage2 输出行寻址。原子写仍在 CK kernel 中，但每个有效 slot 有独占行，不再争写同一 token 输出。`ATOM_MELLUM_MOE_COMBINE=atomic` 可恢复原 AITER 路径。其他模型不走此分支。

限制：只支持上述单卡 BF16 配置；不支持分片 expert、量化、其他激活或 CK one-stage/ksplit，遇到这些配置会明确报错。没有修改 `atom/models/mellum.py`。

## 验证

| 检查 | 结果 |
|---|---|
| 固定 L0 输入、权重和路由，M=1、866；每张卡各 40 次 | 每种情况只有 1 个输出哈希；两张卡哈希相同 |
| 生产服务，同一 A-0000 prompt，16 token，物理 GPU1 | 3/3 输出相同 |
| 同一服务迁至物理 GPU0 | 3/3 输出相同，且与 GPU1 的 16 个 token 相同 |
| A 场景 20×64，生产服务 vs 保存的 HF eager 参考 | 246/1280 = **0.1922**，0/20 整串相同；**未达到 0.90** |
| B 场景 20×64 | 1280/1280 = **1.0000** |

两次服务均在请求结束后正常退出；`pgrep` 无 `atom.entrypoints`，两张物理 GPU 的 VRAM 占用回到 0%。三端口同批复测、C 现场重算、MTP、并发吞吐仍未做。本次只证明所测输入的复现性；不能据此宣称所有 prompt 或调度场景都确定。

原始服务日志与结果在服务器 `/workspace/logs/step0_continuation/`：`server_ordered_det3.log`、`server_ordered_det3_gpu1.log`、`ordered_service_a20_64/summary.json`、`ordered_service_b20_64/summary.json`。`HIP_VISIBLE_DEVICES=0` 映射到物理 GPU1，`=1` 映射到物理 GPU0（以 `rocm-smi` 的显存观察确认）。

## 为什么与 HF 仍有误差

在同一 L0 M=866 捕获数据上，实际 HF 5.2.0 Mellum 参考走 `grouped_mm_experts_forward`，最后 `reshape(...).sum(dim=1)`；重建其输出与捕获文件逐值一致（1,995,264/1,995,264）。这不是运行中的逐专家 Python `index_add_` 循环。旧 ATOM `_atom_moe_hf_order` 是另一条诊断实现，使用逐专家矩阵乘法和 `index_add_`；它既不复刻当前 HF 运算顺序，也不能仅凭 Python 循环或 `index_add_` 的通用文档推断这个具体输入必然确定或不确定。此处没有对该诊断循环做固定输入重复隔离测试。

| L0 对照 | max 绝对误差 | mean 绝对误差 |
|---|---:|---:|
| CK stage1 gated vs HF gated | 0.0625 | 1.118e-4 |
| 同 CK stage2 逐 slot + 定序，CK stage1 输入，最终 vs HF | 0.03125 | 1.689e-4 |
| 同 CK stage2，替换为 HF stage1 gated，最终 vs HF | 0.03125 | 9.134e-5 |
| HF gated + CK stage2 不融合路由权重，按 HF BF16 次序另乘权重、再求和 vs HF | 0.015625 | 2.410e-8 |

因此定序聚合去掉了已观测的竞争写与跨运行差异，但不会把 CK 算术变成 HF bit exact。剩余误差主要来自 stage1 的 BF16 舍入位置及 stage2 路由权重融合时机；最后一行仍有 164/1,995,264 个值不同，不能写成 0 误差。原始分解：`/workspace/logs/step0_continuation/moe_residual_l0.json`；复现实验：`scripts/probe_moe_residual.py`。

## 性能与回退

相同捕获输入、固定路由、完整 MoE 调用的原型微基准：M=1 中位延迟 0.229519→0.432941 ms（**1.886×**）；M=866 为 1.175128→1.430395 ms（**1.217×**）。它不等于生产服务 token/s；新服务吞吐尚未测。需要恢复旧行为时，设置 `ATOM_MELLUM_MOE_COMBINE=atomic` 并重启服务；基线提交为 `07c11d40`。当前分支保留了完整回退路径，未推送。

## 结论

生产路径已接入同 CK stage2 的逐 slot 输出及定序 reduce，固定路由 40 次与双卡服务短样本均复现。Step 0 **仍未通过**：A20 仅 0.1922；后续精度工作应针对 stage1 舍入与 stage2 权重时机，并以现场 HF 参考复测，不能用本次确定性修复替代精度验收。
