# HANDOVER 2026-11-03 06:10 — 最终状态

## 一句话

参考路径（路径2）**已验证为可信基准**；生产路径（路径3）在 B 场景已 1.0000、C 场景 0.77、A 场景 0.27。**Step 0 未通过**，原因是 A 场景的 HF 决策间隔小于算子误差，必须 bit-exact 才能达标。

## 三路径实测（单客户端、干净状态、当前代码）

| 场景 | 路径1 HF | 路径2 参考 | 路径3 生产 |
|---|---|---|---|
| A（20 条，64 token）| 基准 | **1.0000**（20/20）| 0.2727（349/1280，1/20）|
| B（20 条，64 token）| 基准 | **1.0000**（20/20）| **1.0000**（1280/1280，20/20）|
| C（6 条，8 token）| 基准 | 干净样本全一致 | 0.7708（37/48，4/6）|

## 核心解释：决策间隔 vs 算子误差

HF 的 top1-top2 间隔（首 token）：

```
A: [0.125, 1.5, 2.0, 2.0, 2.875, 3.625]  中位 2.0，最小 0.125
B: [5.77, 5.83, 5.96, 6.16, 6.18, 6.27]   中位 6.16
```

A-0006 逐步间隔（前 10 步）：`[2.375, 3.625, 0.625, 0.875, 6.438, 0.625, 0.0, 0.375, 0.375, 0.125]`

**第 6 步间隔 = 0.0（完全并列），第 9 步 0.125。10 步中 2 步落在算子误差带（0.03~0.14）内。**

生产算子与 HF 的差异（L0，`ATOM_PROD_DUMP` 实测）：attention 0.031、MoE 0.14。

因此：token 一翻，后续前缀全错，一致率是存活过程 `(1-p)^k`。A 的 p≈0.2，B 的 p≈0，C 的 p≈0.05——三个分数由此得出。

## 路径2 的干净配置（硬性）

```
ATOM_REFERENCE_MODE=1 ATOM_REFERENCE_RECOMPUTE=1 ATOM_REFERENCE_ATTN4D=1 ATOM_REFERENCE_OWN_CACHE=1
```

**禁止**：`ATTN_CHUNK`、`MOE_BATCHED`、并发客户端、请求中途杀进程、等待中重启服务。

## 已实测排除的方案

| 方案 | 结果 |
|---|---|
| 4-D scores 移植进 flash kernel | 0.586/1280 < 基线 0.740 |
| 只替换 MoE 为参考实现 | 0.1984（无净收益）|
| 参考数学 + 引擎分页 KV | 0.0156 |
| ATTN_CHUNK=1024 | 单块也破坏（1.0→0.35）|
| MOE_BATCHED、请求隔离 | 有害 |

## 若要继续（唯一可行方向）

1. 给生产 attention 加 HF 同序实现：`atom/model_ops/attention_mha.py` 第 555（prefill）/ 395（decode）/ 648（dispatch）行；参考实现见 `reference_mode.py:632`。
2. 生产 MoE 去原子累加：专家权重 = shuffled `cat(gate,up)`（`_unshuffle` 是逆变换），`ReplicatedLinear` 的 gate 权重是**转置**存放的。
3. 达标判据建议分场景给数，并额外报告「HF 间隔低于算子误差带的步数占比」——那是任何实现的理论上限。

## 操作约束（每一条都对应一次数据被毁）

1. 参考服务只能单客户端
2. 不能杀进行中的请求
3. 不能在客户端等待时重启服务
4. 参考文件必须先验真（C-0001 那条 HF 自己都重现不出来）
5. 生产 `blk` hook 与 HF decoder-layer hook 不是同一语义

## 关键提交

```
f6d295d2  4-D attention scores（路径2 达到 1.0000 的转折点）
a373049d  回滚有害的请求隔离改动
18d8fba8  ATOM_PROD_DUMP（路径3 profiling 工具）
82d105f7  A-0006 有完全并列步
0b768ad5  存活模型
```
## 最终补充（07:30，覆盖 06:10 之后的新发现）

### 已被量化排除的生产环节

| 环节 | 实测 | 结论 |
|---|---|---|
| 权重 / embedding / norm 权重 / 路由 | 逐位相同 | 排除 |
| 融合 attention（prefill 与 decode）| 与整行 fp32 softmax 差 **4.9e-4** | 精准，非误差源 |
| RoPE | half-split + theta **5e5**，与 torch 重构差 **0.093**（bf16 ulp=0.125）| 精准 |
| MoE | 与 HF 差 **0.14** | **确认是误差源** |

### 误差的后果已量化

```
HF 决策间隔（A）: 最小 0.125，中位 ~2.0，其中一步恰好 0.0
HF 决策间隔（B）: ~6（数百个 ulp）
```

- 间隔 0.125 的步（A-0003 首 token）：基线选 470（错）→ 换参考 MoE 后选 233（**对**）
- 间隔 2.0 的步（A-0000 首 token）：基线就对
- **MoE 的 0.14 误差 > 0.125 的间隔** ⇒ 这一步必翻；修掉 MoE 就恢复

### 因此的正确施工顺序

1. **修 MoE 的原子累加**（已验证能救回边际决策，自包含）
2. 再查 `rope_cache` 之前对 q/k 的 RMSNorm，以及融合 `qkv_proj`
3. attention 不用动——它已经是精准的

### 交付门槛的诚实建议

A 场景 20 条里存在间隔为 0.0 的步。任何非逐位实现都会在那里做出不同选择，翻转后前缀错误、后续 token 全错。建议验收时：分场景给一致率，并对 A 额外报告「间隔低于算子误差的步数占比」——那是任何实现的理论上限。
## 07:15 追加：最终结论（送验收前请读这一段）

今天把生产路径的每一个环节都单独测了一遍，**没有一个环节有明显缺陷**：

| 环节 | 与精确数学的偏差 |
|---|---|
| 权重 / embedding / 路由 | 0（逐位相同）|
| RMSNorm（q_norm/k_norm）| ~5 个 bf16 ulp（且用的是普通权重、eps=1e-6）|
| RoPE | 1 个 ulp 内（half-split，theta 5e5）|
| attention（prefill 与 decode）| 4.9e-4 |
| MoE | 1.4e-1（原子累加顺序）|

但 L0 attention 输出与 HF 差 **3e-2**（约 30 ulp）。**没有任何单一环节能解释这个差距**——它是每层约 6 个融合阶段的 bf16 舍入，在 28 层上累积的结果。

而 A 场景的 HF 决策间隔最小 **0.125**，且有一条 prompt 的某一步间隔恰好 **0.0**。

### 结论

A 场景的失败**不是可修的单一算子缺陷**，而是**逐阶段舍入预算超过了决策间隔**。因此「A 场景 ≥90% token 一致」这个门槛，**任何非 bit-exact 的实现都达不到**；要做到 bit-exact 就必须放弃让 ATOM 快的融合 kernel。

**这是需要验收方拿着这些数字明确做出的判断。**

### 可立即执行的改进（已验证有效，未做全局收益验证）

1. **MoE 去原子累加**：间隔 0.125 的步在修复后恢复正确（470 → 233）。这是自包含、可验证的一步。
2. B 场景已 1.0000，C 场景 0.77，无需改动。
## 07:30 追加：确定性（一个已验证的真问题，与一致率无关）

### 现象

生产路径**连自己都复现不了**。同一 prompt、greedy、16 token、同一台服务连发 3 次：

```
融合 MoE（默认）: run0 == run1 != run2（第 14 个 token 分叉）
有序 MoE        : run0 == run1 == run2  逐位相同
```

### 成因（已证明）

融合 MoE kernel 用原子累加，不同 run 的求和顺序不同 → 边际决策翻掉。

### 代价（已实测）

```
融合 MoE : 26.32 tok/s（tpot 30.0 ms）
有序 MoE :  9.56 tok/s（tpot 94.8 ms）   ← 2.75× 代价
```

### 建议

1. **确定性必须单独修**：任务要求「greedy 可复现（MTP 开关相同、双卡不变）」，当前生产路径**不满足**，这与是否对齐 HF 无关。
2. `ATOM_PROD_MOE=ref` 只是**证明因果的实验工具**，它的代价（2.75×）不可接受作为产品方案。
3. 正确的修法：保留融合 kernel，把累加改成确定性归约（树形归约 / 按专家顺序串行累加）。这需要改 aiter kernel，Python 层做不到。
4. **不要把确定性修复与一致率修复混为一谈**——前者有明确成因和修法，后者除非完全 bit-exact 无解。

### 数据可复现路径

- 实验脚本：`scripts/det3.py`（3 次同 prompt 比对）、`scripts/perf.py`（吞吐）
- 服务：`ATOM_PROD_MOE=ref` 开关
### 2026-11-03 07:40 (round 92): the production path also fails dual-GPU invariance

The task requires that greedy output not change between the two GPUs. Same prompt (A-0000), same server settings, one request each, only the device differs:

```
GPU0: [233, 233, 37, 26828, 23, 233, 2116, 895, 333, 1948, 302, 1354, 48, 470, 1297, 7912]
GPU1: [233, 233, 37, 26828, 23, 233, 2116, 895, 333, 1948, 302, 1354, 60, 470, 1297, 7912]
                                                                    ^^   ^^
```

They differ at position 12 (48 vs 60) and agree afterwards. So the production path is not invariant across devices either - a second stated requirement that it currently fails.

Same root cause: the fused MoE's atomic accumulation order depends on how blocks are scheduled, and that differs between devices as well as between runs. GPU1's output matches the 'run2' divergence seen in round 90, which is consistent.

## Where this leaves the three task requirements about determinism

| requirement | production path | evidence |
|---|---|---|
| greedy reproducible run to run | **fails** | 3 runs, run2 diverges at token 14 |
| dual-GPU output identical | **fails** | GPU0 vs GPU1 differ at token 12 |
| MTP on/off identical | not measured | no MTP run attempted in this session |

With the ordered MoE substitution the first two become true by construction (the operations are ordered PyTorch calls); the third would need its own measurement.
### 2026-11-03 07:50 (round 93): the ordered MoE fixes both determinism requirements - verified

Same experiment as round 92 (one request per device, same prompt), but with the ordered MoE:

```
GPU0: [233, 233, 37, 26828, 23, 233, 2116, 895, 333, 1948, 302, 1354, 60, 392, 37495, 497]
GPU1: [233, 233, 37, 26828, 23, 233, 2116, 895, 333, 1948, 302, 1354, 60, 392, 37495, 497]
identical: True
```

Combined with round 90 (three consecutive runs identical), the ordered accumulation satisfies:

| requirement | fused MoE | ordered MoE |
|---|---|---|
| greedy reproducible run to run | fails (token 14) | **passes** (3/3 identical) |
| dual-GPU output identical | fails (token 12) | **passes** |
| MTP on/off identical | not measured | not measured |

So two of the three determinism requirements in the task can be met today, by changing only the MoE accumulation order. The cost is 2.75x throughput with this particular substitution (26.3 -> 9.6 tok/s), which is why the right fix is a deterministic reduction inside the fused kernel rather than replacing it.

## Session summary for the 08:00 handover

**Not achieved**: three-path unification; scenario A agreement (production 0.07-0.33, unstable, against a 1.0000 reference).

**Achieved and verified**:
1. the reference path reproduces HF exactly (A 20/20, B 20/20, C exact on every clean sample), reproduced on the final tree
2. every production stage measured against exact math: weights/norm-weights/routing bit-identical, RMSNorm ~5 ulp, RoPE 1 ulp (half-split, theta 5e5), attention 4.9e-4, MoE 0.14
3. the reason A cannot pass: accumulated rounding (6 fused stages x 28 layers) exceeds A's decision margin, whose minimum is 0.125 with one step at exactly 0.0
4. seven candidate fixes measured, all ineffective at scenario level
5. the production nondeterminism traced to the MoE atomics and **fixed** by ordered accumulation, satisfying run-to-run and dual-GPU reproducibility
### 2026-11-03 07:35 (round 94): MTP cannot be measured here - attempted, not assumed

Tried to enable the third determinism requirement (MTP on/off identical) by starting the server with `--method mtp`:

```
server came up DOWN
log: self.drafter = EagleProposer(self.config, self.device, self)
     model_class = resolve_obj_by_qualname(support_eagle_model_arch_dict[draft_model_hf_config.architectures[0]])
```

`--method mtp` resolves to the Eagle process, which requires a `--draft-model`; the Mellum checkpoint at /data/models/Mellum2.1-12B-A2.5B-Thinking does not ship one and none was found in the environment.

So the status of that requirement is **not measured**, for a concrete reason (missing draft model), not because it was skipped.

To measure it later: obtain the Mellum draft checkpoint, then run `scripts/one_run.py` against a server with and without `--method mtp --draft-model <path>` and compare the token ids - the same procedure used for the run-to-run and dual-GPU checks.
## 关键提交（完整，按时间顺序）

```
f6d295d2  4-D attention scores（路径2 达到 1.0000 的转折点）
78da84c3  KV 不重分配清零
a373049d  回滚有害的请求隔离改动
18d8fba8  ATOM_PROD_DUMP（路径3 profiling 工具）
23816f10  ATTN_CHUNK 回归（测量作废声明）
1ac4ee10  参考路径在 C 上其实是对的（此前的 C 失败全是测量污染）
1dfee0b3  参考 MoE 能救回 0.125 间隔的步
2d37d8f0  HF 同序 eager attention（含 KV 布局文档）；证明融合 attention 精准
4f223ef3  attention 在 prefill/decode 两个阶段都被排除
fb868420  RoPE 精确验证（half-split，theta 5e5）
679c116e  RMSNorm 精准；无单一 bug，是累积舍入
53a348b8  确定性代价 2.75×（26.3 → 9.6 tok/s）
534d2567  生产路径双卡不一致（第 12 token）
ac888007  有序 MoE 同时修复跨 run 与双卡可复现（已验证）
82bf3b6b  MTP 无法测量（缺 draft 模型，附失败日志）
6a825c9d  报告去重与最终整理
```
## 工作产物在哪里（交付位置说明）

所有提交都在**服务器本地**，没有推送到远端。

| 仓库 | 路径 | 分支 | 最新提交 |
|---|---|---|---|
| 工作树 | `/workspace/ATOM-tritonmoe` | `dsh/moe-determinism` | `bbe48a6a` |
| 共享 | `/workspace/ATOM` | `mellum-step0` | `313e9f5b` |

`git remote -v` 指向 `https://github.com/ROCm/ATOM.git`（上游开源仓库），两个分支都**没有 upstream**。

**我没有推送**，因为没有推送授权，而且推送到上游开源仓库不是本次任务的授权范围。

取走产物：

```
# 在服务器上打包（推荐）
cd /workspace/ATOM-tritonmoe && git bundle create /tmp/mellum-step0.bundle --all
# 或直接拷走文档
scp -P 2211 root@43.180.254.230:/workspace/ATOM-tritonmoe/{BRIEF-0800.md,DEBUG_GUIDE.md,REPORT-STEP0-2026-11-03.md,HANDOVER.md,RESULTS.md} .
```
## 2026-11-03 09:43 continuation: production MoE source audit

Read `PRODUCTION_MOE_AUDIT.md` before changing MoE. The production bf16 top-8 path is Opus sorting → CK stage1 → CK stage2; CK stage2 uses `AtomicAdd` directly to bf16 `moe_buf`. Fixed-router, fixed-stage1 tests (40 repeats per GPU) locate nondeterminism in stage2 rather than routing. Profiler measured 3 kernels per M=1 `fused_moe` call and 4 per M=866 call, excluding routing. The exact buffer shapes, source lines, precision comparison and raw log paths are in the audit. The original L0 `0.14 / 8.2e-4` statistic was not recreated in this isolated capture. Next engineering step is a same-configuration per-slot stage2 output plus ordered reduction, followed by full production service correctness/throughput checks. The mistaken PyTorch eager-default change was reverted in `b3dc6092`. No push; Step 0 remains unpassed.

## 2026-11-03 10:01 continuation: stage2 combine isolated under same CK GEMM

The requested per-slot stage2 + ordered reduce experiment is complete. See `SLOT_STAGE2_ORDERED_REDUCE.md` and `scripts/probe_moe_slot_stage2.py`. With fixed sorting and fixed stage1, both GPUs showed original CK stage2 40/40 distinct outputs, but per-slot CK stage2 and ordered reduction 1/40 distinct output; ordered hashes were equal across GPUs. The original and per-slot calls used the same compiled CK GEMM kernel name/tile, with runtime topk and row indexing changed to separate slots. At M=866, vs HF mean fell from about `2.186e-4` to `1.689e-4`, max from `0.0625` to `0.03125`. This is a diagnostic implementation, not a production-service switch; full token correctness and throughput remain untested. Step 0 remains unpassed.
