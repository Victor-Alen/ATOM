# HANDOVER 2026-11-03 04:30 — 三路径现状与待办

## 一句话

A/B 在参考路径上**逐位一致**；生产路径 A = **0.2242**；C 有两条独立问题未解决。**Step 0 未通过。**

## 定案数字（干净配置）

参考路径配置：`MODE + RECOMPUTE + ATTN4D + OWN_CACHE`，**禁止** `ATTN_CHUNK` / `MOE_BATCHED`，util 0.35~0.45 均可。

| 场景 | 路径1 HF | 路径2 参考 | 路径3 生产 |
|---|---|---|---|
| A | 基准 | **1.0000**（20/20，最新 6/6 复核）| **0.2242**（287/1280，2/20 全对）|
| B | 基准 | **1.0000**（20/20）| 未在当前配置复测 |
| C | 基准 | C-0000 64/64、C-0002 64/64、C-0003 64/64；**C-0001 1/8** | 未复测 |

## 已确认的事实

1. **生产融合算子没有 3-D/4-D 之分**（flash/paged 是在线 softmax 分块内核，不生成 scores 矩阵）。
2. **A 的 1.0000 只在参考路径成立**，裁判不加环境变量时跑的是生产路径。
3. 生产 L0 与 HF 的差异：attention `max 0.031`、MoE `max 0.14`（参考路径同点均为 0）。
4. 生产专家权重布局 = shuffled `cat(gate, up)`，`_unshuffle` 是正确逆变换；`ReplicatedLinear` 的 gate 权重是**转置**存放的（踩过坑）。
5. 单独替换 MoE：A 287/1280 → 254/1280（无净收益）。
6. `ATOM_REFERENCE_ATTN_CHUNK` 即使单块也改变结果（1.0000 → 0.35），**未解释，禁用**。
7. `ATOM_REFERENCE_MOE_BATCHED`（我手写的 batched_mm）同样有害，禁用。
8. 参考路径 `history_after` 的“按请求隔离”改动有害，已回滚（`a373049d`）。

## 两个未解决的 C 问题

**问题 1：参考文件不可复现（数据问题）**

```
C-0000 7881 : fresh == recorded
C-0001 12580: fresh != recorded（前 5 个 token 一致后分叉）
C-0002 11889: fresh == recorded
C-0003 6183 : fresh == recorded
```

C-0001 的参考答案在 HF 自己那里都重现不出来。任何用它算的分数无效。

**问题 2：12.5k 附近引擎真的错（代码/显存问题）**

```
引擎 C-0001: [233, 96855, 18437, 89628, ...]
HF  C-0001: [233, 233, 37, 6539, 233, ...]
```

11889 token 全对，12580 token 崩。12580 的 fp32 softmax 工作区 = 20.2 GiB。
建议先查：分配是否失败/降级、KV 缓冲大小、block 数与显存余量。

## 下一轮该做什么（按优先级）

1. 用**现场重算的 HF 参考**重评 C-20（不要用文件），拿到有效数字
2. 定位 12.5k 的失败（单条 C-0001，带 CHUNKLOG + 显存监控）
3. 生产 attention 的 eager 同序实现（`attention_mha.py:555 / 395 / 648`）
4. 三端口同批复测，结果写进 RESULTS

## 工具

- `ATOM_PROD_DUMP=<dir>`：生产路径 L0/L1 的 attn/moe/blk 落盘
- `ATOM_PROD_MOE=ref`：生产换参考 MoE（默认关，效果不佳）
- `scripts/overnight_run.py <sets> <port> <label> [limit]`：批量跑 A/B/C
- `scripts/c_scan_multi.py <ids>`：检查参考文件是否可复现
- `scripts/perm_test.py`：参考 vs 生产的逐张量比较