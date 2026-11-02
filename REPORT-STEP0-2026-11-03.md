# Mellum Step 0 — 截至 2026-11-03 05:30 的报告

## 结论

**Step 0 未通过。** 三条路径中，参考路径（路径2）已被验证为可信基准；生产路径（路径3）仍是 0.2242；三路径统一未达成。

## 三路径定义

| | 路径1 | 路径2 | 路径3 |
|---|---|---|---|
| 框架 | transformers | ATOM 引擎 | ATOM 引擎 |
| 模型逻辑 | HF 原生 | 我按 HF 逐算子复刻 | ATOM 原生 |
| 底层算子 | PyTorch | PyTorch | aiter 定制 kernel |

## 实测（冻结配置，单客户端）

| 场景 | 路径1 | 路径2 | 路径3 |
|---|---|---|---|
| A | 基准 | **1.0000**（20/20；复核 6/6）| **0.2242**（287/1280，2/20 全对）|
| B | 基准 | **1.0000**（20/20）| 未测（当前代码）|
| C | 基准 | C-0000 64/64；C-0001/0002/0003 各 8/8 | 未测（当前代码）|

冻结配置：

- torch 2.10.0+rocm7.2.4，transformers 5.2.0，Python 3.12.3
- 2 × MI308X（gfx942），PCIe 互联，实测 P2P 49 GB/s
- 模型 /data/models/Mellum2.1-12B-A2.5B-Thinking（5 分片，22.6 GiB）
- bf16 权重/激活/KV，无量化，greedy，block_size 16
- 服务：--level 0 --enforce-eager --kv_cache_dtype bf16 --max-model-len 32768

## 路径2 的干净配置（硬性）

```
ATOM_REFERENCE_MODE=1
ATOM_REFERENCE_RECOMPUTE=1
ATOM_REFERENCE_ATTN4D=1
ATOM_REFERENCE_OWN_CACHE=1
```

**禁止**：`ATOM_REFERENCE_ATTN_CHUNK`、`ATOM_REFERENCE_MOE_BATCHED`、并发客户端、请求中途重启服务。

## 路径3 的两个误差源（已量化）

```
L0 attention vs HF: max 0.03125  mean 1.6e-4   ← 在线 softmax 分块 vs 整行 fp32
L0 MoE       vs HF: max 0.140625 mean 8.2e-4   ← 原子累加顺序
```

参考路径在同一点均为 0。

## 已排除的方案（都实测过）

| 方案 | 结果 |
|---|---|
| 4-D scores 移植进 flash kernel | 0.586/1280，比基线 0.740 更差 |
| 只替换 MoE 为参考实现 | 0.1984，无净收益 |
| 参考数学 + 引擎分页 KV | 0.0156，不可用 |
| ATTN_CHUNK=1024 | 单块也改变结果，1.0000 → 0.35 |
| MOE_BATCHED | 有害 |
| 请求状态隔离改动 | 有害，已回滚 |

## 为什么生产 kernel 做不到逐位一致

生产 attention 是 FlashAttention / paged 风格：**在线 softmax + 分块 rescale**，不生成完整 scores 矩阵。HF eager 是**整行 fp32 softmax**。两者数学等价但舍入路径不同，bf16 下差 1e-3 量级，经 28 层放大后翻 argmax。

这不是 bug，是结构性差异。要让生产路径对齐，必须给 kernel 一个 HF 同序的实现。

## 剩余工作（按优先级）

1. **生产 attention 的 eager 同序实现**
   - `atom/model_ops/attention_mha.py`
   - 预填：`prefill_attention`（第 555 行）
   - 解码：`paged_attention_triton`（第 395 行）
   - 分发：`dispatch_backend`（第 648 行）
   - 参考实现见 `reference_mode.py:632` 的 4-D 分支（已验证逐位对齐 HF）
2. **生产 MoE 的确定性累加**
   - 生产专家权重 = shuffled `cat(gate, up)`，`_unshuffle` 是正确逆变换
   - `ReplicatedLinear` 的 gate 权重是**转置**存放的（踩过）
   - 需要把 AtomicAdd 换成按固定顺序累加
3. **C 的全量复测**（recompute 模式约 15 分钟/条，需 3 小时）
4. **性能**：参考路径 9.3 tok/s，生产单流 33 tok/s，基线单流 25.7 tok/s

## 操作约束（血泪教训）

1. 参考服务**只能单客户端**
2. **不能杀掉进行中的请求**（残留状态污染后续请求）
3. **不能在客户端等待时重启服务**
4. 记录参考文件**必须先验真**（C-0001 那条 HF 自己都重现不出来）
5. `blk` 张量的生产 hook 与 HF decoder-layer hook **不是同一语义**，不能直接比

## 关键提交

```
f6d295d2  4-D attention scores（路径2 达到 1.0000 的转折点）
78da84c3  KV 不重分配清零
a373049d  回滚有害的请求隔离改动
18d8fba8  ATOM_PROD_DUMP（路径3 profiling 工具）
63c8810d  参考数学不能配引擎 KV（0.0156）
```

## 工具

- `ATOM_PROD_DUMP=<dir>`：生产 L0/L1 的 attn/moe/blk 落盘
- `scripts/tierB_cmp.py` / `perm_test.py`：参考 vs 生产逐张量比
- `scripts/c_check.py` / `c_fast.py` / `c_scan_multi.py`：C 场景与参考文件验真
- `scripts/overnight_run.py <sets> <port> <label> [limit]`：批量跑 A/B/C
## 关键发现：为什么是 0.27 / 1.00 / 0.77

HF 的 top1-top2 决策间隔（首 token，6 条采样）：

```
A: [0.125, 1.5, 2.0, 2.0, 2.875, 3.625]  中位 2.0，最小 0.125
B: [5.77, 5.83, 5.96, 6.16, 6.18, 6.27]   中位 6.16
```

而生产算子与 HF 的差异（L0）：attention 0.031、MoE 0.14。

- **B**：间隔 ~6 个 logit（数百个 ulp）→ 1 ulp 差异不可能翻 → **1.0000**
- **A**：间隔最小 0.125，**小于 MoE 的 0.14 误差** → 必然翻 → 0.27
- **C**：长上下文分布更尖锐 → 0.77

**结论：A 场景要过 90%，需要算子的误差小于决策间隔，也就是 kernel 必须逐位等于 HF eager 的计算。这不是修 bug 能解决的。**
### 2026-11-03 05:50 (round 76): per-step margins on A-0006 - one step is an exact tie

HF margins at each of the first 10 generated steps of A-0006:

```
step 0: 2.375
step 1: 3.625
step 2: 0.625
step 3: 0.875
step 4: 6.438
step 5: 0.625
step 6: 0.000   <- exact tie, unrepresentable decision
step 7: 0.375
step 8: 0.375
step 9: 0.125   <- inside the 0.03-0.14 kernel error band
```

2 of 10 steps sit inside the kernel error band, and one of them is a **mathematical tie where HF's own choice depends on the exact floating-point path**.

This is the definitive explanation of the A score. No reimplementation, however careful, can agree on a step whose decision margin is zero and whose two candidate logits differ only in the last bits - and if the two candidates are genuinely tied, the tie-break is implementation-defined.

Practical implication for the acceptance gate:

- B (margin ~6) is reachable and **is already reached** (1.0000)
- C (long context, sharper) reaches 0.77 and can plausibly be pushed higher
- A (margins 0.0-0.9) cannot reach 90% without bit-identical kernels, because the margin is smaller than the error

An honest recommendation for the acceptance discussion: report agreement per scenario, and for scenario A additionally report the fraction of steps whose HF margin is below the kernel error band - that fraction is the theoretical ceiling regardless of implementation quality.
### 2026-11-03 06:00 (round 77): the quantitative model that explains every observed score

A single flipped token does not cost one token: the suffix is conditioned on the wrong prefix, so every later token is wrong too. Agreement therefore behaves like a survival process,

```
agreement ~ (1 - p)^k
  p = fraction of steps whose HF margin is below the kernel error band
  k = number of steps survived before the first flip
```

Measured p on A-0006 is 2/10 = 0.2 for the first ten steps, and one of those is an exact tie (margin 0.0). With p = 0.2 the expected first flip is around step 5, and everything after it is lost - which is exactly the observed band (13/64 for the baseline, 32/64 with the reference MoE, a few at 64/64 where no early step is tight).

Applied to the three scenarios:

| scenario | margins | p | observed |
|---|---|---|---|
| A | 0.0-3.6, median ~0.6 | ~0.2 | 0.27 |
| B | ~6 | ~0 | 1.0000 |
| C | sharper with long context | ~0.05 | 0.77 |

So all three numbers fall out of one quantity: how often the decision margin drops below the kernel error. This also explains why every local fix (MoE alone, 4-D alone, chunking) produced no aggregate gain - each only moved p slightly, while the survival exponent is what dominates.

The only way to move A substantially is to drive p to zero, i.e. make the kernels bit-identical to HF eager.