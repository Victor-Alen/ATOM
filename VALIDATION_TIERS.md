# 三层正确性验证体系（2026-11-03）

## 三个独立层次

| 层 | 对比 | 验证什么 | 状态 |
|---|---|---|---|
| **A** | HF PyTorch vs ATOM Python Reference | 模型计算逻辑 | 通过（A/B 1.0000）|
| **B** | ATOM Reference vs ATOM Production | 底层算子 | 失败（生产 A 0.2242），正在 profiling |
| **C** | Scheduler / KV / Engine | 推理引擎执行 | 未开始 |

关键澄清：A 层通过不证明 B 层（参考与生产数值算法不同），也不证明 C 层（HF generate 不是 Serving Scheduler）。

## B 层 profiling（当前重点）

生产可挂钩点（atom/models/mellum.py）：

| 行 | 类 | 钩子 |
|---|---|---|
| 86 | MellumSparseMoeBlock | 输出 = moe_out，输入 = post_attn_norm |
| 143 | MellumAttention | 输出 = attn_out |
| 268 | MellumDecoderLayer | 输入 = 残差流，输出 = blk_out |

实现：在模型初始化后按 env ATOM_PROD_DUMP 注册 forward hook，把 L0/L1 的 attn/moe/blk 存成 .pt，
再与 HF 同语义张量逐位比较。

判据：

| 结果 | 结论 |
|---|---|
| L0 attn 不同、moe 相同 | attention kernel 数值算法差异 |
| L0 attn 相同、moe 不同 | MoE kernel（AtomicAdd）差异 |
| L0 全不同 | 输入/布局/元数据不一致 |
| L0 全同、L1 不同 | 逐层累积 |

## 已知事实（勿重复验证）

1. 生产融合算子没有 3-D/4-D 之分（flash/paged 是在线 softmax 分块内核）
2. 把 4-D 搬进生产无效（另一 agent：全程 586/1280 < 基线 740/1280）
3. 生产 A = 0.2242（冻结配置，2/20 全对）
4. 参考路径 A/B = 1.0000

## C 层测试清单

单请求（已完成）、并发 2 请求、continuous batching、prefix caching、preemption、
KV block 分配释放残留、混合 prefill/decode。

注意：ATOM_REFERENCE_MODE=1 只替换 ModelRunner 的算子执行路径（mellum.py:511/531），
调度器、KV 管理、批处理仍是生产那套，所以参考路径能验证部分调度行为，但不能验证算子正确性。

## 下一步

1. 给生产挂 hook，对 A-0006 抓 L0 的 attn/moe/blk 与 HF 比，定位第一个不同张量
2. attn 不同 -> 在 dispatch_backend 加 eager 分支
3. moe 不同 -> 替换 aiter 的 AtomicAdd 为固定次序累加
4. B 层达标后再做 C 层