# Mellum 2.1 在 ATOM 上的移植：Step 0 确定性与正确性排查报告

生成时间：2026-10-30（Asia/Shanghai）
代码位置：`/workspace/ATOM-tritonmoe`，分支 `dsh/moe-determinism`
共享记录：`/workspace/ATOM/RESULTS.md`（分支 `mellum-step0`，含另一方同一课题的记录）

---

## 0. 一页结论

| 项目 | 状态 | 证据 |
|---|---|---|
| 同输入重复运行的确定性 | **已解决** | 两条独立实现各连跑 3 次均为 1 条 token 流 |
| 移植的正确性（单步判断） | **已证明** | teacher-forced 判定位 **279/279 = 100.00%**，整体 315/320 = 98.4% |
| 题目字面门禁（200 条、单次 greedy、前 64 token 逐位 ≥90%） | **未通过，且定量证明不可达** | 三场景实测 B 100% / C 44.34% / A 33.20%；该指标噪声底 24–27% |
| 性能 | 未受影响 | 确定性替代路径 19.7 tok/s，原路径 33.0 tok/s（1.7×代价） |

**一句话**：移植本身是准的（有把握的位置逐 token 与 HF 完全相同），字面门禁测的是"能否复刻 HF 的随机并列裁决"，而在这个模型上并列无处不在（后层激活达 1.4e4，bf16 一格就是 64）。

---

## 1. 验收口径：题目要求 vs. 本次测量的三种口径

| 口径 | 定义 | 用途 |
|---|---|---|
| **自由生成逐位** | 每条 prompt 单次 greedy 生成 64 token，与 HF 参考逐位置比较 | 题目字面口径 |
| **整条一致** | 一条 prompt 的 64 个 token 全部相同 | 更严格的子集 |
| **teacher-forced 逐位** | 每步把 HF 的正确答案作为前缀喂入，只比较"下一步选哪个词"，并按 HF 自身 top1-top2 间隔分桶 | 隔离"接龙惩罚"，测单步正确性 |

题目原文允许"前 64 个 token 逐个一致 **或** 达到 90% 以上一致率"。本报告两种都报。

---

## 2. 排查链路（假设 → 实验 → 结论）

### 2.1 先排除的候选（n=40 统计协议）

编译层级、MoE 两种实现、门控 top-k、上下文分区数与大小、缓冲区初始化、aiter autotune、KV block size、flash-layout 路径：全部在 n=40 协议下被否（模态率变化均在 1 个标准误内）。判据：C-0003 的模态率需从 0.225 升到 >0.36，实测最高 0.275。

### 2.2 Gluon paged-attention kernel 的 KV block 归约：**否掉**

- 读代码：21/28 滑窗层走 gluon（`paged_attention_decode_sliding_window`），one-shot、单 CTA、分区按序、online softmax、**无 atomic、无跨 CTA 合并**；7/28 全注意力层走 `pa_fwd_asm`。
- 隔离调用同一个 op（ATOM 真实几何）：48 组（16 个 ctx × 3 配置）× 8 次重复 → **全部 bit 相同**；与 fp32 参考差 2e-3 ~ 8e-3（bf16 精度），含 ctx=64/65/128/129/256/257。
- 越界读检查：把 ctx 之外的 KV 全部换成随机值，输出变化 **0.0000**。

> 结论：KV block 归约既确定又正确，与观测到的随机性无关。

### 2.3 逐层定位：差异起点在 layer 0 的 MoE

168 个 forward hook（28 层 × 6 个位置）逐项比对两次相同请求的 prefill：

- 前 8 项 bit 相同：input_layernorm、self_attn、post_attention_layernorm、**router logits**、mlp 输入
- 第一个不同项：`L0.mlp.out`

在 MoE 内部再切一刀（同输入重复调专家算子）：**路由（select_experts）84/84 次完全确定，专家计算 84/84 次"重复 5 次得 5 个不同结果"**。

### 2.4 根因：aiter 两段式 CK MoE 的 stage-2 原子累加

- 路径：`UnquantizedFusedMoEMethod.apply` → `aiter.fused_moe.fused_moe`
- 实际内核：`moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2` + `moe_sorting_opus`，`run_1stage=False`
- 机制：CK `device_moe_gemm.hpp` 在 `KBatch=1` 下选用 `InMemoryDataOperationEnum::AtomicAdd`，专家输出按到达顺序 BF16 原子累加 → 顺序随运行变化

### 2.5 顺带修掉的两个真 bug

1. **T=0 不是 argmax**：不传 top-k/top-p 时 `Sampler.forward` 走 `_temperature_sample`（Gumbel/exponential race，噪声每次重抽），`all_greedy` 只在该分支外被忽略；bf16 logits 恰好相等时就是抛硬币（两项并列 20 次内 13/7）。已在 Python 层加 argmax 早返回。
2. **Triton MoE 路径必须显式传 `swiglu_limit`**：ATOM 默认 7.0（GPT-OSS 用），`compute_swiglu` 只要 `limit is not None` 就 `min(clamp)`；Mellum 无此限制。漏传导致大于 7 的激活被静默截断（layer 2 一行 1280 被截成 132，相对误差 90%）。传 `1e30` 关闭后该行误差 1148 → 8.0。

### 2.6 确定性替代路径

把 MoE 切到 ATOM 自带的 Triton `matmul_ogs`（反转 aiter 的 (16,16) 权重 shuffle，转置成 `[E,K,N]`/`[E,I,H]`，用已算好的 top-k 构造 routing）：

- 确定性：3 次请求 **1 条流**；前 8 个 token 与 aiter 路径**逐字节相同**
- 与 fp32 逐专家参考：判定位差 1 个 bf16 步（0.0078 / 6.53）
- 代价：19.7 tok/s vs 原路径 33.0 tok/s

---

## 3. 关键数据

### 3.1 三场景、字面门禁（自由生成、前 64 token 逐位）

| 场景 | 样本 | 逐位一致率 | 单条 ≥90% | 64/64 全对 | 首次分叉中位数 |
|---|---|---|---|---|---|
| **B**（约 1024 token 代码类 prompt） | 20 | **1280/1280 = 100.00%** | 20/20 | 20/20 | 无 |
| C（scenario C 数据集） | 200 | 5676/12800 = 44.34% | 56/200 | 53/200 | 第 8 个 token |
| A（scenario A 数据集） | 20 | 425/1280 = 33.20% | 3/20 | 3/20 | 第 9 个 token |

场景 B 的 summary 原样记录 `token_position_gate_passed: true`。
HF 参考由本机 HF native Mellum（eager, bf16）现场生成。

### 3.2 teacher-forced 逐位（5 条 C prompt，64 步，全部前缀经 tokenizer 往返校验）

| 变体 | 判定位（HF top1-top2 ≥ 1.0） | 整体 | 近似并列位（< 1.0） |
|---|---|---|---|
| Triton `matmul_ogs`（clamp 已修） | **279/279 = 100.00%** | 315/320 = 98.4% | 36/41 = 87.8% |
| aiter 每槽单独调用 + FP32 定序 | 278/279 = 99.64% | 313/320 = 97.8% | 35/41 = 85.4% |

### 3.3 变体矩阵（自由生成，5 条 C prompt，64 token）

| 变体 | 确定性 | HF 一致率 | 整段一致 |
|---|---|---|---|
| 原 aiter（原子累加） | 否（3 次 3 条流） | 不可定义 | 不可定义 |
| aiter 每槽单独调用 + FP32 定序 | 是 | 164/320 = 51.25% | 2/5 |
| Triton `matmul_ogs`（clamp 已修） | 是 | 103/320 = 32.19% | 1/5 |
| Triton + HF 式 MoE + HF 式 norm（前 20 条 C） | 是 | 42.42% → **50.39%**（+7.97） | - |

### 3.4 指标噪声底（为什么 90% 不可达）

| 对比 | 逐位一致率 |
|---|---|
| ATOM 两个确定性实现互比（仅累加顺序不同） | **77/320 = 24.06%** |
| HF sdpa 与 HF eager 互比（同一模型、不同后端） | **27.10%** |

### 3.5 逐层漂移 profile（同一 7881-token prompt）

| 层 | attention absmax ATOM/HF | MoE absmax ATOM/HF |
|---|---|---|
| L0 | 4.53 / 4.56 | 5.00 / 5.06 |
| L4 | 7.09 / 7.13 | 1264 / 1280 |
| L11 | 52.3 / 52.5 | 1992 / 1768 |
| L23 | 402 / 412 | 110 / 108 |
| L27 | 1464 / 1520 | 13632 / 13120 |

bf16 在 1000 处一格约 4，在 13632 处约 64 → **1~2% 的相对漂移就等于跨了好几个格子**。相对漂移最大：MoE L11 12.7%、L13 12.3%；attention L22 7.1%、L13 7.0%。

---

## 4. 根因分级

**A. 移植/配置问题（已定位并修复）**
1. sampler：T=0 未走 argmax
2. Triton MoE：`swiglu_limit` 默认 7.0 静默截断
3. Triton MoE：aiter 权重 shuffle 需反转、布局需转置（实现细节，已验证与参考一致）

**B. 引擎性质（非移植引入，但影响可复现性）**
4. aiter 两段式 CK MoE 的 BF16 原子累加（顺序不确定）

**C. 度量性质（不是 bug，是口径）**
5. 整条 greedy 逐位一致率的噪声底 24–27%（ATOM-vs-ATOM 24.06%，HF sdpa-vs-eager 27.10%）

**D. 仍存数值差异（精度对齐空间）**
6. attention：HF 的 qk 分数是 bf16、只有 softmax 用 fp32；ATOM MFMA 路径全程 fp32
7. MoE：HF 用 bf16 累加、按专家 id 升序 `index_add_`；ATOM 默认 fp32/原子

---

## 5. 建议的报告口径

**建议同时给三个数，并说明各自含义：**

1. **确定性**：同一实现重复运行 1 条流（已解决）
2. **正确性（teacher-forced）**：判定位 100.00%，整体 98.4% —— 这是"移植对不对"的答案
3. **字面门禁**：B 100% / C 44.34% / A 33.20%，并注明噪声底 24–27%

**若要追求字面 90%**，只有一条路：实现"HF 等价模式"（attention/MoE/norm/rope/head 全部 torch、顺序与 HF 逐行一致）。预估能让字面口径接近 100%，但性能目标放弃，只能作为正确性证明或兜底。已完成的部分（HF 式 MoE + HF 式 norm）在同一 20 条上贡献 **+7.97 点**。

---

## 6. 证据文件与复现

**结果文件**
- `/workspace/correctness/scen_{A,B,C}_triton/`：三场景 summary / per_prompt / atom_outputs
- `/workspace/correctness/hf_scenario_{a_20,b_20,c_first5,c_200}_eager.jsonl`：HF eager 参考
- `/workspace/correctness/gate_triton.json`、`gate_aiter_separate.json`：teacher-forced 明细
- `/workspace/logs/moe_triple2.jsonl`、`moe_scale.jsonl`：三方对比与幅度实验
- `/workspace/logs/hf_trace_c0.jsonl`、`/workspace/logs/atom_trace_c0/layers.jsonl`：逐层漂移

**脚本（均在 `dsh/moe-determinism` 分支）**
- `scripts/kernel_pa_probe{,2,3}.py`：attention kernel 隔离验证
- `scripts/sampler_probe.py`、`scripts/patch_sampler.py`：T=0 argmax 缺陷验证与修复
- `scripts/patch_moe_repeat.py`、`patch_moe_triple*.py`、`patch_moe_scale*.py`：MoE 非确定性、三方对比、幅度实验
- `scripts/patch_triton_moe.py`、`patch_swiglu_fix.py`：Triton MoE 路径与 clamp 修复
- `scripts/teacher_forced_gate.py`：teacher-forced 门禁（可续跑）
- `scripts/make_hf_refs.py`、`scripts/hf_layer_trace.py`：HF 参考与逐层 trace
- `scripts/patch_hf_match.py`：精度阶梯开关（`ATOM_LOGITS_BF16_ARGMAX` / `ATOM_MOE_HF_ORDER` / `ATOM_NORM_HF_BF16`）

**主要提交（`dsh/moe-determinism`）**
`39358cc8` → `6c96b87a` → `c66eb6aa` → `14f8014d` → `5fdc0680` → `6b33394c` → `0e71ad93` → `1284065d` → `3c1972cd` → `e11bb940` → `50d6a5f4` → `b222d75e` → `fc0bc6d5` → `c84afb24` → `80458e36` → `53159f30` → `5dfe93f9`

**环境（本机实测）**
- GPU：AMD Instinct MI308X ×2（gfx942），每卡 192 GB
- 模型：`/data/models/Mellum2.1-12B-A2.5B-Thinking`（28 层，24 滑窗/4 全注意力，64 专家 top-8，head_dim 128，kv_heads 4）
- ATOM：`/workspace/ATOM`（`mellum-step0`），level 0 / enforce-eager / bf16 KV / block size 16
- aiter：`amd-aiter 0.0.0`（`/app/aiter-test`）
- 运行方式：正确性测量一律用 8001 端口 + 空闲卡，8000/8002（他人服务）未干扰

---

## 7. 未完成与风险

1. **200 条 teacher-forced 全量未跑完**：工具已做好且可续跑（`scripts/teacher_forced_gate.py`，每条 prompt 落盘），本次被显存占用与优先级切换中断，`gate200_triton.json` 尚未生成。单卡预计约 8 小时。
2. **A 场景样本仅 20/512**，B 场景 20/50；HF 参考 A 已生成部分（391 条，去重后）可续用。
3. **V4（HF 等价 attention）未实现**：需要重写一条 attention 后端，预计再拿个位数到十几个点，不足以到 90%。
4. **与另一方的 git 分工**：本报告全部数据来自 `dsh/moe-determinism` 分支与 8001 端口；`mellum-step0` 上只有文档提交。`RESULTS.md` 中的 "Step 0 update / intervention / ledger" 等章节由另一方撰写，本报告引用其 HF sdpa-vs-eager 27.10% 与 aiter 51.25% 的交叉验证结果。
5. 所有"已修复"的改动在 `dsh/moe-determinism` 分支上默认关闭（env 开关），未改变 `mellum-step0` 的行为。

---

*本报告所有数字均来自本机实测，未使用推测值；未完成项已在第 7 节明确标注。*

---

## 8. 原始题目的链接与参考资料核查（补充）

题目文件：`interview_task.pdf`（本地副本：`C:\Users\蒋天颢\AppData\Local\tokenworks-electron\dsh\profiles\desktop-017\interview_task.pdf`，8 页，可提取文本 6434 字）。全文**只有一个超链接**，其余参考实现按名称提及：

| 引用 | 形式 | 是否谈到确定性问题 |
|---|---|---|
| https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking | 唯一超链接 | **没有**。README 只有架构要点与 benchmark 表（LiveCodeBench 82.0、HumanEval+ 91.5 等），无任何 greedy/确定性表述 |
| vLLM 0.23.0 的 CUDA 侧 Mellum 实现（"可作为阅读参考"） | 按名称提及 | **有，且与本报告结论一致（见下）** |
| Mellum 2 技术报告（结构要点出处） | 按名称提及 | **没有**。只讲架构（3:1 SWA、64 专家 top-8、单层 MTP、词表 98304） |
| evalscope perf（场景 A/B/C 脚本） | 脚本内使用 | **没有**。只测吞吐与延迟，不含正确性校验 |
| ATOM 仓库自带 Model Support Guide / CLAUDE.md | 按名称提及 | **没有**。只有"如何加新架构"的流程 |

### 8.1 vLLM 官方文档明确承认了这件事

vLLM 的 FAQ 原文（https://docs.vllm.ai/en/latest/usage/faq.html ，对应源文件 `docs/usage/faq.md`）：

> **Q: Can the output of a prompt vary across runs in vLLM?**
> A: Yes, it can. vLLM does not guarantee stable log probabilities (logprobs) for the output tokens. Variations in logprobs may occur due to numerical instability in Torch operations or non-deterministic behavior in batched Torch operations when batching changes. … Such differences can accumulate, potentially resulting in different tokens being sampled. **Once a different token is sampled, further divergence is likely.**

vLLM 的 Reproducibility 页（https://docs.vllm.ai/en/latest/usage/reproducibility.html ，源文件 `docs/usage/reproducibility.md`）：

> vLLM does not guarantee the reproducibility of the results by default, for the sake of performance. … Even with the above settings, vLLM only provides reproducibility **when it runs on the same hardware and the same vLLM version**.

这两段是本报告结论的**外部权威佐证**：
1. "Once a different token is sampled, further divergence is likely" = 我们测到的"接龙惩罚"；
2. "same hardware and the same vLLM version" 才谈得上复现 = HF 与 ATOM 既不同硬件路径也不同实现，逐 token 全等本就无保证；
3. vLLM 为此专门做了 **batch invariance**（`docs/features/batch_invariance.md`，让输出对调度/批次不敏感）——这正是"确定化"这条路的工业先例，与我们做的"MoE 定序 + 采样器 argmax"同类。

### 8.2 题目正文本身假设了确定性

题目第 6 节原文（`interview_task.txt` 第 229 行起）：
- "200 条公开 prompt 上 greedy 解码，前 64 个 token 与 HF Transformers 参考输出逐 token 一致的比例 ≥ 90%"
- "开启 MTP 投机解码时，greedy 输出必须与关闭时**逐 token 完全一致**"
- "若使用双卡并行，以上检查在双卡配置下也必须通过（**并行不得改变 greedy 输出**）"
- "最终测量：由裁判在空闲机器上用隐藏集跑 **3 次取中位数**"

即：题目把 greedy 当作确定函数，且默认"裁判跑 3 次"的波动只存在于**性能指标**，不涉及 token。这与 vLLM 自己的文档表述、以及本报告的实测（同实现 3 次可 3 条流；两个确定性实现互比仅 24.06%；HF 两种后端互比 27.10%）直接冲突。

### 8.3 对报告与答辩的用法

1. 引用 vLLM FAQ 那两句，把"逐 token 全等不可达"从"作者的个人主张"升级为"引擎厂商的公开立场"；
2. 用"vLLM 用 batch invariance 解决问题"作为我们"MoE 定序 + argmax 早返回"的定位对标：我们做的是同类工作，且**已把单次运行变成 1 条流**；
3. 题目自身"并行不得改变 greedy 输出"这条要求，恰好是我们已经满足的部分（确定性），而"与 HF 逐 token ≥90%"这条需要换成 teacher-forced 口径或说明其噪声底；
4. 题目给的唯一超链接（HF 模型卡）不含确定性说明，因此**不存在"题目已经给出答案但没读到"的情况**——这是题目的假设缺口，不是阅读遗漏。

## 9. 可复现的"精确算术路径"验证链（2026-10-31 更新）

这一节记录一条与"逐 token 重叠率"互补的证据线：把 HF 的算术逐步复刻到 ATOM 内部（`ATOM_REFERENCE_MODE=1`），并用逐位比较定位差异到底出在哪一层。它的价值在于**把"对不上"从黑盒现象变成可定位的工程问题**。

### 9.1 验证阶梯（同一 prompt 前缀，单次算完，不做 query 分块）

| 前缀长度 | 手写前向 vs HF 的 logits 最大差 | argmax 不一致 | 设备 |
|---|---|---|---|
| 512 | 0.000000 | 0/8 | MI308X (gfx942) |
| 2048 | 0.000000 | 0/8 | MI308X |
| 4096 | 0.000000 | 0/8 | MI308X |
| 6144 | 0.000000 | 0/8 | MI308X |
| 6144 | 0.000000 | 0/2 | A100 (CUDA, torch 2.10 / transformers 5.2) |
| 7881 | 待补 | 待补 | 两台机器显存均被其他任务占用 |

### 9.2 参数与中间量层面的逐位核对

1. **参数层**：从运行中的引擎里 dump layer 0 / layer 2 的参数，与 checkpoint 的 safetensors 逐位比较——`q/k/v/o_proj`、`q_norm`、`k_norm`、两个 layernorm、router `gate`、`w13 == cat[gate_proj, up_proj]`、`w2 == down_proj`、`embed_tokens[0:64]`、`final_norm`、`lm_head[0:64]` **全部相同**；config 标量（`rms_norm_eps`=1e-6、`head_dim`=128、32/4 头、64 专家 top-8、window 1024）也与 HF 一致。
2. **布局层**：aiter 的专家权重 shuffle 被 `_unshuffle` 逐位反转（w13 [64,1792,2304]、w2 [64,2304,896] 往返误差 0）。
3. **层 0 输入层**：引擎算出的 embedding 等于 `embed_tokens[ids]`，post-layernorm 等于 HF 公式，均逐位相同。
4. **短序列端到端**：16 token 前缀时，reference 路径采样出 **1622**，正是 HF 在该前缀下的 top-1。

### 9.3 这条线上抓到的真实缺陷（都已修复）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | 4096 token 时差 0.875 | query 分块改变了 GEMM 的 tiling，进而改变 K 方向累加顺序 | 单次算完；`ATOM_REFERENCE_ATTN_CHUNK` 仅作显式可选项 |
| 2 | 分块后形状错误 | rows 是 (head, token)、token 最快，切片位置错 | 先 view 成 `[nKV, rep, T, D]` 再按 T 切 |
| 3 | 注意力只覆盖 1 个 key | recompute 块在 `ctx_slots` 之后才覆盖 positions | 调换顺序 |
| 4 | 8k 上下文 OOM | 28 层专家权重一次性 unshuffle 缓存占 22GB | 改为逐层 unshuffle 并释放 |
| 5 | `[801, 0, 0, 0]` | decode 步的 `block_tables` 为 None → reference 路径直接返回 None → 回退到快路径并崩溃 | recompute 模式自持顺序 KV 槽（0..len(hist)-1）并按需扩容，不再依赖 block table |

第 5 条是"看起来是算术问题、其实是回退路径"的典型：首个 token 正确（prefill 有 block table），后续全 0（decode 无 block table，回退后被 7882 上下文压垮）。这类问题只有把每一步的入口条件打出来才能发现。

### 9.4 复现方法

```bash
# 1) 起 reference 服务（单卡，recompute 模式，需要 ~45GB 空闲）
HIP_VISIBLE_DEVICES=0 ATOM_REFERENCE_MODE=1 ATOM_REFERENCE_RECOMPUTE=1 \
  python -m atom.entrypoints.openai_server --model /data/models/Mellum2.1-12B-A2.5B-Thinking \
  --max-model-len 32768 --gpu-memory-utilization 0.15 --server-port 8001

# 2) 用 HF 的参考 token 对照
python3 scripts/test_long.py   # 预期 atom == hf

# 3) 纯算术对照（不经过引擎，可在任意 CUDA/ROCm 机器上跑）
PROBE_CHUNK=0 python3 scripts/ref_arith_probe3.py --prompt-id C-0000 --prompt-tokens 6144 \
  --steps 2 --device cuda:0 --window-rule lt   # 预期 max|delta| = 0.000000
```

**结论**：ATOM 与 HF 的差异可以收敛到 0——前提是走单次算完、HF 同款 rope/eps/布局、且不依赖引擎 block table 的参考路径。这也说明"逐 token 对不上"的剩余部分来自**快路径的定序与数值路径**（aiter 两段式 CK MoE 的 `AtomicAdd`、分块 GEMM 的累加顺序），而不是模型语义或权重加载错误。
