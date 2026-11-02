# OVERNIGHT PLAN — 2026-11-03 (目标：08:00 前三条路径统一)

## 当前状态（01:10）

| 项 | 状态 |
|---|---|
| GPUs | 已全部接管，其他 agent 的服务已终止 |
| 8001 | 参考路径（MODE+RECOMPUTE+ATTN4D+OWN_CACHE），A/B/C 正在跑 |
| 8002 | 生产基线（无环境变量），A/C 正在跑 |
| 参考路径 A | **1.0000**（3/3，revert 后确认）|
| 参考路径 C | 之前 148/148（另一 agent 记录），本次重跑中 |
| 生产基线 A | 0.2292（早前 6 条）|
| 生产基线 C | 跑动中（C-0036 11/64、C-0037 56/64 等波动）|

## 关键结论（已确定，不再重复验证）

1. **生产融合算子没有 3-D/4-D 之分**：`pa_fwd_asm` / `pa_decode_gluon` 是在线 softmax 分块内核，不生成 scores 矩阵。把 4-D 搬进去无效（另一 agent 实测：全程 586/1280，仅预填 618/1280，均低于基线 740/1280；仅解码 759/1280 略好）。
2. **真正的差异是数值算法**：HF eager = 整行 fp32 softmax；flash/paged = 在线 softmax + 分块 rescale。舍入路径不同 → 1e-3 级差异 → 被 MoE 与残差放大 → 翻 argmax。
3. **MoE 的 AtomicAdd 必须换成固定次序累加**：按 expert 排序 → grouped GEMM → 按 perm 顺序 index_add。并要镜像 HF 的 dispatch（prefill 用 grouped_mm、decode 用 batched_mm）。
4. **参考路径的 request-isolation 改动有害已回滚**（`a373049d`）。

## 生产候选实现（8003）——待执行

### 修改点

文件：`atom/model_ops/attention_mha.py`

| 行 | 函数 | 动作 |
|---|---|---|
| 555 | `prefill_attention` | 加 eager 分支 |
| 395 | `paged_attention_triton` | 加 eager 分支（decode，1 行 query，代价极低）|
| 648 | `dispatch_backend` | 读 `ATOM_PROD_ATTN=eager` 时返回 eager |

### eager 实现（复用已验证的 4-D 代码）

```python
def eager_attention(q, k, v, positions, window, scale):
    # q: [T, nH, D]  k/v: [ctx, nKV, D]（从 paged KV 按 slot 拼出）
    q4 = q.transpose(0, 1).unsqueeze(0)                    # [1, nH, T, D]
    k4 = k.transpose(0, 1).repeat_interleave(nH // nKV, 0).unsqueeze(0)
    v4 = v.transpose(0, 1).repeat_interleave(nH // nKV, 0).unsqueeze(0)
    sc = torch.matmul(q4, k4.transpose(2, 3)) * scale
    neg = ...  # (qi <= ki) 且 (qi - ki) < window
    sc = sc.masked_fill(neg, float("-inf"))
    pr = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(pr, v4).squeeze(0).transpose(0, 1)  # [T, nH*D]
```

**这正是参考路径里已验证逐位对齐 HF 的写法**（`reference_mode.py:632` 的 4-D 分支）。

### MoE 修改点

文件：`atom/models/layers/moe.py`（或实际 aiter MoE 封装）

```python
perm = torch.argsort(sel.reshape(-1), stable=True)
grouped = F.grouped_mm(x[perm], w13.transpose(-2,-1), offs=offs)
...
out.index_add_(0, token_ids[perm], out_g * w[perm])
```

## 验证矩阵（三端口，同一批 prompt）

| | 8001 参考 | 8002 生产基线 | 8003 生产候选 |
|---|---|---|---|
| A | 期望 1.0 | 0.23（已测）| 待测 |
| B | 期望 1.0 | 待测 | 待测 |
| C | 期望 ≥0.9 | 跑动中 | 待测 |

判据：三端口用同一份 `hf_scenario_*_eager.jsonl` + `hf_scenario_a_20_fresh.json`，64 token greedy。

## 执行顺序

1. 等 8001 的 A/B/C 跑完（约 45 分钟）→ 确认参考路径 3 场景全 ≥90%
2. 等 8002 的 A/C 跑完 → 生产基线完整数字
3. 释放 GPU0，起 8003（候选）→ 先单条验证不崩，再跑 A/C
4. 若候选达标 → 合并进主仓默认路径（去掉环境变量）
5. 若候选不达标 → 回滚，记录原因

## 禁止事项

- 不得把参考路径的成绩当作生产路径成绩
- 不得在未验证的情况下提交（round 54 的教训）
- 所有结论必须写到 RESULTS.md 并带提交号
