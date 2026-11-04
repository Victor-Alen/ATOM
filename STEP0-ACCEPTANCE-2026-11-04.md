> 范围更正：本报告是逐token一致性阶段记录，不代表HumanEval、MTP开启或正式双卡并行均已验收。完整复核见 STEP0-FIVE-GATES-RECHECK-2026-11-04.md。

# Mellum Step 0 实测验收 — 2026-11-04

当前候选通过本次 Step 0 正确性门槛（各项 token 一致率 ≥90%）。以下结果全部从原始 API token_ids 独立复算，不使用健康检查代替生成验证。

冻结计算代码：`a787ed595fd4a0c50b9c54aff2a98de3961c26b8`，工作树 `/workspace/ATOM-memory-safe`，分支 `dsh/mellum-memory-safe`。

| 验证 | 请求与输出长度 | 与 HF 相同的位置 | 一致率 | 整段一致请求 |
|---|---:|---:|---:|---:|
| A20 串行 | 20×64 | 1280/1280 | 100.00% | 20/20 |
| B20 串行 | 20×64 | 1280/1280 | 100.00% | 20/20 |
| A32 / PCI22 | 32×64 | 2048/2048 | 100.00% | 32/32 |
| A32 / PCI30 | 32×64 | 2048/2048 | 100.00% | 32/32 |
| C200 双卡泊松 | 200×64 | 12800/12800 | 100.00% | 200/200 |

两张卡的同一组 A32 输出逐条相同：32/32。

## 验证范围与配置

- 原生 ATOM Mellum 模型、真实权重、生产调度器与分页 KV Cache；没有参考模型 forward、保存张量注入或 reference mode。
- 运算后端是 HF 同序 GPU 算子、grouped/batched experts，以及逐请求形状的 GEMM/RMSNorm。**不是原 CK fast path 已修复**，该候选旁路了 CK experts。
- BF16 权重/激活/KV；TP=1；level=0、enforce-eager；prefix cache 关闭；max_model_len=32768；gpu_memory_utilization=0.45。
- A32 是 A20 循环组成的 32 请求混合负载，每张卡各测一轮，不冒充完整 A512 数据集验收。
- C200 使用全部 200 条冻结 eager 参考，每条 64 token；总到达率 0.30 请求/秒，seed42，轮流发往两台 TP1 服务，200 个 HTTP worker，不使用上一轮的 32 在途请求限制。
- C 实际 HTTP 峰值在途数 21；最大到达延迟 0.001197 秒，中位数 0.000145 秒；回放耗时 665.6 秒。
- 参考文件是已核对 tokenizer 输入、并与新 HF A10/C10 输出交叠核对的冻结 eager 基准；本轮没有重新生成全部 200 条 HF 参考。参考与计算源码 SHA256 在审计 JSON 中。

## 哪些实验支持修复

1. 原候选 d05b171b 在两张卡串行 A20/B20/C200×64 均100%，跨卡相同。A0002/C0003 各40次串行重放也100%。但混合 A32 只有196/2048 token、1/32整段一致；只生成首 token 也只有17/32，说明串行通过不能推出并发通过。
2. 固定批次实验60组、375请求：2/4/8/16/32路各重复5次，输出在同一组成下稳定；8路起出现首 token 分叉。已记录的活动写入 slot 无重复。这不等于排除所有 KV 问题，但没有复现随机 race。
3. 相同输入行的独立 QKV 重放，M=3738 与16384的输出相差1个BF16值，且逐位复现了真实服务两侧结果。实验没有调度器、共享请求状态或KV Cache，直接证明批形状导致的计算差异存在。
4. 保留逐请求 GEMM 形状、关闭分块 prefill 后，A32 从196/2048提高到1888/2048。剩余分叉在较晚 decode。
5. A0003 第22个输出位置的 L4 RMSNorm，输入、residual、相加结果相同，M=1/32的 FP32 方差分别为0.2897353172302246和0.28973525762557983，产生17个BF16差异。独立归约实验逐位复现。逐请求 RMSNorm 后，两卡 A32 均2048/2048。
6. 相同 token 前缀下，A0002 第29个选词：单请求 token409/596 logits=27.0/26.875，批量=26.75/26.875，选词翻转。A0003 第30个选词：单请求302/2677均26.375，批量2677升至26.5。记录把中间变化、最终logits和选词连接起来；不把分叉后的不同前缀混作同输入数值误差。
7. 43c5be1f 的单卡 C 泊松测试前32条正确，随后在C0049（16019-token prompt）发生FP32 softmax OOM，**该轮失败**。新候选保留完整QK/PV GEMM，仅分head执行softmax；6种固定输入形状均逐位相同，16019形状额外分配峰值由61.18GiB降到22.94GiB。本轮C0049与完整C200已实际重跑，结果见上表。

## 限制

本次是正确性验收，不是原始 A512 高并发吞吐/C200×256-token 性能验收；没有证明吞吐、延迟达到生产目标。逐请求算子与 dense eager attention 有性能代价。两卡分别运行TP1服务，不代表TP2或MTP已验证。

当前关闭 chunked prefill、max_num_batched_tokens=16384，因此本配置接受的 prompt 长度上限为16384；本次C最大16019，满足范围。不能据此宣称32768-token prompt已可用。

## 复现与证据

运行前显式指定获分配GPU，例如：

```bash
cd /workspace/ATOM-memory-safe
ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=1 MELLUM_SERVER_PORT=8005 ATOM_MELLUM_REQUEST_GEMM=1 bash scripts/start_mellum_hf_arithmetic.sh
# 第二台：HIP_VISIBLE_DEVICES=0，MELLUM_SERVER_PORT=8007
```

完整原始日志：`/workspace/logs/step0_causal_boundary/memory_safe_v3_acceptance`。独立审计：`/workspace/logs/step0_causal_boundary/memory_safe_v3_acceptance/final_independent_audit.json`。双卡C回放脚本：`/workspace/logs/step0_causal_boundary/c_poisson_two_gpu.py`。实际GPU映射、命令行、环境与源码哈希见各gpu目录 runtime_manifest.json。

因果实验原始张量保留在远端 `batch_invariance_v1` 与 `request_decode_trace_v1`；交付证据包包含小型重放指标、选词链、失败记录与本轮原始API响应，不包含模型权重。
