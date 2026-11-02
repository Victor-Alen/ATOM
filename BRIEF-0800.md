# 8 点简报 — Mellum Step 0（2026-11-03 07:40）

## 一句话

**Step 0 未通过。** 参考路径完全正确；生产路径 B 达标、C 0.78、A 0.27，且 A 的差距不是可修的单一 bug。

## 三路径实测（单客户端、干净状态、当前代码树）

| 场景 | 路径1 HF | 路径2 参考 | 路径3 生产 |
|---|---|---|---|
| A（20×64 token）| 基准 | **1.0000** | 0.07–0.33（不稳定）|
| B（20×64 token）| 基准 | **1.0000** | **1.0000** |
| C（20×8 token）| 基准 | 逐位精确 | 0.7812 |

## 今晚证明的三件事

**1. 每个生产环节单独都不差**

```
权重 / embedding / norm 权重 / 路由 : 逐位相同
RMSNorm                            : ~5 个 bf16 ulp
RoPE（half-split, theta 5e5）      : 1 ulp 内
attention（prefill 与 decode）      : 4.9e-4
MoE                                : 0.14   ← 唯一明显偏差
```

**2. A 过不了是因为累积舍入超过决策间隔**

- 每层 6 个融合阶段 × 28 层的 bf16 舍入 → L0 与 HF 差 3e-2
- A 的 HF 决策间隔最小 **0.125**，其中一步恰好 **0.0**
- **预算 > 间隔 ⇒ 必然翻 token ⇒ 后续前缀错误 ⇒ 整段崩**
- 因此 A ≥90% 需要 bit-exact kernel，不是修 bug

**3. 生产路径本身不确定（独立于 HF 对齐）**

```
跨 run : 3 次运行，第 3 次在 token 14 分叉
双卡    : GPU0 与 GPU1 在 token 12 不同
根因    : 融合 MoE 的原子累加顺序
```

**修法已验证**：改用有序累加后，跨 run 与双卡都逐位一致；代价是 26.3 → 9.6 tok/s（2.75×）。正确修法是保留 kernel、把累加改成确定性归约（需改 aiter）。

## 已实测无效的七种修复

4-D 移植（740→586）、单换 MoE（A 287→254；C 125→127）、eager attention（0.2727→0.0547）、参考数学+引擎 KV（0.0156）、ATTN_CHUNK（1.0→0.35）、MOE_BATCHED（1.0→0.45）、请求隔离（已回滚）。

## 三端口同批复测（目标要求的协议，已执行）

三台服务**同时**运行（8001 参考 + 8002 生产基线在 GPU0，8003 生产+有序MoE 在 GPU1），
同一批三条 A prompt、64 greedy token：

```
端口 8001  参考路径       : 192/192 = 1.0000   exact 3/3
端口 8002  生产基线       :  42/192 = 0.2188   exact 0/3
端口 8003  生产+有序MoE   :  39/192 = 0.2031   exact 0/3
```

参考路径在两台生产服务抢占同一张卡时仍是 192/192——确定性在资源竞争下成立（运算有序）。

## 目标指定的方案：已实现，已测试，无效

目标要求「生产路径接入 eager 同序 attention + HF MoE dispatch」。两个都实现了，同时启用测了：

```
生产基线                          :  42/192 = 0.2188
只换 eager attention              :  13/192 = 0.0677
只换 HF MoE                       :  39/192 = 0.2031
两者同时（目标指定配置）          :  12/192 = 0.0625   ← 最低
```

**指定组合比基线低、比任何单项也低。**

原因：attention 本来就准（4.9e-4），换掉它几乎不减少误差，只改变舍入路径；MoE 替换
同样改变它参与的所有位置的舍入。两次替换把轨迹推向不同方向，换个方向不等于靠近 HF。

**唯一能达到 HF 精确一致的配置仍是完整参考路径**（端到端有序 PyTorch 运算），慢 3 倍。

## 建议的下一步（按优先级）

1. **先修 MoE 确定性**（自包含、已验证、满足任务的两条确定性要求）
2. 再决定 A 的口径：要么接受 ≥90% 在 A 上不可达（附间隔统计），要么投入 kernel 级 bit-exact 改造
3. MTP 一致性需要 draft checkpoint（当前环境没有，已尝试并记录失败日志）

## 文档索引

| 文件 | 内容 |
|---|---|
| 本文件 | 2 分钟读完的状态 |
| `REPORT-STEP0-2026-11-03.md` | 完整结论与因果链（532 行）|
| `HANDOVER.md` | 交接：要做什么、代价、坑（256 行）|
| `RESULTS.md` | 全部实验原始数据与提交号（2540 行）|

## 复现命令

```
# 参考路径（A/B 应为 1.0000）
ATOM_REFERENCE_MODE=1 ATOM_REFERENCE_RECOMPUTE=1 ATOM_REFERENCE_ATTN4D=1 ATOM_REFERENCE_OWN_CACHE=1 \
  python3 -m atom.entrypoints.openai_server --model /data/models/Mellum2.1-12B-A2.5B-Thinking \
  --level 0 --enforce-eager --kv_cache_dtype bf16 --max-model-len 32768 --server-port 8003 \
  --no-enable_prefix_caching

python3 scripts/overnight_run.py A 8003 <label>     # A/B/C 批量
python3 scripts/one_run.py                         # 单次输出（确定性检查）
python3 scripts/perf.py                            # 吞吐
python3 scripts/det3.py                            # 3 次同 prompt 比对
```
