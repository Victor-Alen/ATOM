# 任务书：Mellum Step 0 三路径统一（续作版）

编写时间：2026-11-03 08:40（Asia/Shanghai）
编写依据：2026-11-02 22:00 — 2026-11-03 08:00 的实际执行与实测数据，无推测内容
交接仓库：`/workspace/ATOM-tritonmoe`（分支 `dsh/moe-determinism`，HEAD `3d3319f3`）、`/workspace/ATOM`（分支 `mellum-step0`，HEAD `bac3be1a`）

---

## 一、任务目标（原文）

> 在 2026-11-03 08:00 前完成 Mellum Step 0 的三路径统一：参考路径（A/B/C 全部 ≥90% 且最好 100%）、生产路径把 eager 同序 attention + HF MoE dispatch 接入并达到同样的 ≥90%，三端口（8001 参考/8002 生产基线/8003 生产候选）同批复测，全部结果与提交号写入 RESULTS.md 与 HANDOVER.md。不得虚构，未验证的必须标注。

**截止时点结论：未通过。** 参考路径达标，生产路径未达标。

---

## 二、三路径定义

| | 路径1 | 路径2 | 路径3 |
|---|---|---|---|
| 名称 | HF 原生 | ATOM 参考实现 | ATOM 生产实现 |
| 框架 | transformers | ATOM 引擎 | ATOM 引擎 |
| 模型逻辑 | HF 官方 modeling | 按 HF 逐算子复刻 | ATOM 原生 |
| 底层算子 | PyTorch | PyTorch | aiter 定制 kernel |

三层验证的职责边界（详见 `VALIDATION_TIERS.md`）：
- A 层（HF vs 路径2）= 验证模型计算逻辑
- B 层（路径2 vs 路径3）= 验证底层算子
- C 层（调度/KV/引擎）= 需独立测试，**A 层通过不能推出 C 层**

---

## 三、环境（冻结配置）

- 硬件：2 × AMD MI308X（gfx942），每卡 192 GiB，80 CU；**xGMI 单链路互联**（amd-smi：XGMI / 1 hop / weight 15；RCCL topo dump 有 <xgmi count="1">），实测 P2P 带宽 50.5 GB/s（1 GB 传输），56 KiB 小传输 2.0 GB/s（延迟受限）
- 软件：Python 3.12.3、`torch 2.10.0+rocm7.2.4.git3d3aa833`（hip 7.2.53211）、`transformers 5.2.0`
- 模型：`/data/models/Mellum2.1-12B-A2.5B-Thinking`（MellumForCausalLM，28 层，32 Q / 4 KV head，head_dim 128，hidden 2304，vocab 98304，sliding_window 1024，64 experts top-8，rms_norm_eps 1e-6，max_position 131072，5 分片 22.6 GiB）
- 数据类型：bf16 权重/激活/KV，无量化
- 服务参数（三端口统一）：

```
--level 0 --enforce-eager --kv_cache_dtype bf16 --max-model-len 32768
--gpu-memory-utilization 0.45 --host 127.0.0.1 --no-enable_prefix_caching
```

- 评测集：`/workspace/correctness/hf_scenario_{a_20,b_20,c_200}_eager.jsonl`
  - A：20 条，512–4096 token 真实 prompt，64 greedy token
  - B：20 条，同一条 1049 token prompt，64 greedy token
  - C：200 条，4k–15k token 长 prompt，8 greedy token（本次用前 20 条）

---

## 四、当前实测状态（交付树，单客户端，干净状态）

| 场景 | 路径1 HF | 路径2 参考 | 路径3 生产 |
|---|---|---|---|
| A（20×64） | 基准 | **1.0000**（今天复核 14/14，896/896 token）| 0.14–0.33（跨 run 波动）|
| B（20×64） | 基准 | **1.0000**（20/20）| **1.0000**（20/20）|
| C（20×8） | 基准 | 每个干净样本逐位精确 | 0.7812（125/160，12/20 全对）|

三端口同批复测（同批 3 条 A prompt）：

```
端口 8001 参考路径        : 192/192 = 1.0000   exact 3/3
端口 8002 生产基线        :  42/192 = 0.2188   exact 0/3
端口 8003 生产+有序MoE    :  39/192 = 0.2031   exact 0/3
```

---

## 五、已完成并验证的工作

### 5.1 参考路径可用（这是全部结论的基准）

干净配置（**四项必须齐全**）：

```
ATOM_REFERENCE_MODE=1
ATOM_REFERENCE_RECOMPUTE=1
ATOM_REFERENCE_ATTN4D=1
ATOM_REFERENCE_OWN_CACHE=1
ATOM_REFERENCE_MODEL_DIR=/data/models/Mellum2.1-12B-A2.5B-Thinking
```

禁止项（都会破坏结果）：`ATOM_REFERENCE_ATTN_CHUNK`、`ATOM_REFERENCE_MOE_BATCHED`、并发客户端、请求进行中杀进程、等待中重启服务。

### 5.2 逐环节偏差已量化（工具：`ATOM_PROD_DUMP`）

| 环节 | 与精确数学的偏差 | 结论 |
|---|---|---|
| 权重 / embedding / norm 权重 / 路由 | 0 | 逐位相同 |
| RMSNorm（q_norm / k_norm）| ~5 个 bf16 ulp | 精准，非误差源 |
| RoPE | 1 ulp 内 | 精准（**half-split，theta 5e5**）|
| attention（prefill 与 decode）| 4.9e-4 | 精准，**不是误差源** |
| MoE | 1.4e-1 | 唯一明显偏差 |

> 早期报告把 L0 的 3e-2 归因于"在线 softmax 分块"，该归因**已撤回**——实测只差 4.9e-4。

### 5.3 A 场景不可达的因果链（已闭环验证）

1. 每层约 6 个融合阶段（qkv_proj、norm、rope、attention、o_proj、MoE），每阶段几个 ulp
2. 28 层累积 → L0 attention 输出与 HF 差 3e-2
3. A 场景 HF 的 top1-top2 决策间隔：首 token 最小 0.125，中位约 2.0；A-0006 前 10 步中**第 6 步间隔恰好 0.0**、第 9 步 0.125
4. 直接验证：间隔 0.125 的步，基线选 470（错），去掉 MoE 的 0.14 误差后选 233（对）；间隔 2.0 的步基线本来就对
5. 结论：**舍入预算 > 决策余量 ⇒ 翻转必然；一翻，后续前缀全错**

### 5.4 八种局部修复实测无效

| 修复 | 场景 | 基线 | 修复后 |
|---|---|---|---|
| 4-D scores 移植进 flash kernel | A | 740/1280 | 586/1280 |
| 只换 MoE 为参考实现 | A | 287/1280 | 254/1280 |
| 只换 MoE 为参考实现 | C | 125/160 | 127/160 |
| HF 同序 eager attention | A | 0.2727 | 0.0547 |
| **eager attention + HF MoE（任务指定）** | A | 0.2188 | **0.0625** |
| 参考数学 + 引擎分页 KV | — | — | 0.0156（坏）|
| ATTN_CHUNK=1024 | 参考 A | 1.0000 | 0.35 |
| MOE_BATCHED | 参考 A | 1.0000 | 0.45 |
| 请求状态隔离改动 | 参考 A | 1.0000 | 0.016（已回滚）|

### 5.5 找到并验证了一个真实缺陷（未合入）

**现象**：生产路径跨 run 不确定、跨 GPU 不一致

```
跨 run : 3 次运行，第 3 次在 token 14 分叉
双卡   : GPU0 与 GPU1 在 token 12 不同（48 vs 60）
```

**根因**：融合 MoE 的原子累加顺序随 run / 设备变化

**已验证修法**：改用有序累加（`ATOM_PROD_MOE=ref`）→ 3/3 逐位相同、双卡一致

**代价**：26.32 → 9.56 tok/s（**2.75×**，tpot 30.0 → 94.8 ms）

---

## 六、剩余任务（按优先级，可直接执行）

### T1 合入 MoE 确定性修复（唯一已验证有效的改动）

- **目标**：greedy 跨 run 与跨 GPU 可复现（任务明确要求）
- **现状**：`ATOM_PROD_MOE=ref` 分支已在 `atom/models/mellum.py` 的 `MellumSparseMoeBlock.forward` 中实现并验证，默认关闭
- **两个方案，任选**：
  - **方案 A（1 小时）**：把该分支设为默认。已验证有效，代价 2.75× 吞吐
  - **方案 B（半天，推荐）**：只在 decode 阶段走有序累加，prefill 保留融合 kernel；需实测代价与确定性
- **验收**：
  - `python3 scripts/det3.py` → `identical: True`
  - 双卡各起一次 `scripts/one_run.py` → token 列表相同
  - 记录吞吐损失
- **风险**：方案 A 会显著降低生产吞吐，需与性能负责人确认

### T2 MTP 一致性测量（当前无法进行）

- **阻塞原因**：`--method mtp` 走 Eagle 路线，需要 `--draft-model`，环境中无该 checkpoint（已尝试，失败日志在 `/workspace/logs/overnight/mtp.log`）
- **需要你的输入**：Mellum 的 draft checkpoint 路径
- **方法**：同一 prompt 分别对「开 MTP」与「关 MTP」的服务各发一次，比对 token id（脚本 `scripts/one_run.py`）

### T3 A 场景口径决策（需要你拍板）

- **选项 1：接受 A 的 ≥90% 不可达**
  - 交付：分场景报告一致率 + 「间隔低于算子误差的步数占比」（理论上限）
  - 工具：`scripts/margin.py`（首 token 间隔）、`scripts/stepmargin.py`（逐步间隔），已可用
- **选项 2：kernel 级 bit-exact 改造**
  - 范围：融合 qkv_proj、norm+rope、attention、residual、MoE 全部按 HF 顺序重写
  - 代价：性能回到参考路径水平（9.3 tok/s 量级），且是数天工程量
  - 需要明确：是否放弃融合 kernel

### T4 C 场景从 0.7812 提到 0.90（20 条基准）

- **现状**：125/160，12/20 全对；失败多为差 3–5 个 token，不是整段崩
- **已试无效**：单换 MoE 只 +0.013（125→127）
- **建议做法**：
  1. 对 8 条失败样本用 `ATOM_PROD_DUMP` 抓 L0/L1，找到首次分叉位置
  2. 用 `scripts/stepmargin.py` 量该位置的 HF 间隔
  3. 若间隔 > 误差仍分叉 → 那是真正可修的 bug；若间隔 < 误差 → 归入 T3 口径问题
- **验收**：C 20 条 ≥0.90

### T5 全量 200 条验收复测

- **注意事项**：
  - **不能用现成的 C 参考文件**：C-0001 那条 HF 自己都复现不出来（前 5 token 一致后分叉），任何用它算的分数无效
  - 参考必须**现场重算**（或先逐条验真）
  - 参考路径 recompute 模式约 30–60 秒/条（长 prompt 更久），200 条需数小时，建议分批
- **验收**：200 条逐条给出路径2/路径3 的一致率与提交号

### T6 三端口复测脚本化

- 目标里要求「三端口同批复测」，已手工做过一次；建议固化为脚本
- 注意显存：三台同时起，每台 `--gpu-memory-utilization 0.22`（两台在 GPU0、一台在 GPU1 可行，已验证）

---

## 七、未验证事项（必须如实标注）

| 事项 | 状态 | 原因 |
|---|---|---|
| MTP 开关输出一致 | **未测** | 缺 draft checkpoint |
| 200 条全量验收 | **未做** | 时间不足；A 用过 20 条、B 20 条、C 20 条 |
| 32 路并发吞吐 | 未在当前代码树复测 | 历史值 638 tok/s 是 32 路聚合，非单流 |
| 前缀缓存 / 抢占 / 连续批处理 | **未测** | 属于 C 层（调度），本次未覆盖 |

---

## 八、操作约束（每一条都对应一次数据被毁，务必遵守）

1. **参考服务只能单客户端**——并发会污染模块级状态，返回 `(801,0,0,0,0)`
2. **不能杀掉进行中的请求**——残留状态会让后续几个请求都错
3. **不能在客户端等待时重启服务**
4. **启动新服务前确认 `pgrep -cf atom.entrypoints` 为 0**——否则端口被旧服务占着，`curl /health` 照样返回 200，测的是旧服务
5. **参考文件必须先验真**——用 HF 现场重算同一条比对记录值

以上 5 条已在本次执行中各踩一次，详见 `DEBUG_GUIDE.md`。

---

## 九、交付物清单

| 文件 | 用途 | 行数 |
|---|---|---|
| `BRIEF-0800.md` | 2 分钟看结论 | ~110 |
| `DEBUG_GUIDE.md` | 下次怎么查（含坑与死路）| ~120 |
| `REPORT-STEP0-2026-11-03.md` | 完整结论与因果链 | ~560 |
| `HANDOVER.md` | 交接：做什么、代价、坑、交付位置 | ~290 |
| `RESULTS.md` | 全部原始数据与提交号 | ~2600 |
| `VALIDATION_TIERS.md` | 三层验证方法论 | 54 |

取走产物：

```bash
cd /workspace/ATOM-tritonmoe && git bundle create /tmp/mellum-step0.bundle --all
# 或
scp -P 2211 root@43.180.254.230:/workspace/ATOM-tritonmoe/{BRIEF-0800.md,DEBUG_GUIDE.md,REPORT-STEP0-2026-11-03.md,HANDOVER.md,RESULTS.md} .
```

**未推送**：两个仓库都无 upstream，远端为 `https://github.com/ROCm/ATOM.git`（上游开源仓库），无推送授权。

---

## 十、需要你决定的三个问题

1. **A 场景口径**：接受不可达（选项1）还是投入 kernel 级改造（选项2）？
2. **确定性修复优先级**：方案 A（立即，代价 2.75×）还是方案 B（半天，只改 decode）？
3. **MTP**：能否提供 draft checkpoint 路径？否则该项只能标注未测。
