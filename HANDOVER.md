# HANDOVER — Mellum 2.1 在 ATOM 上的正确性（Step 0）工程交接

> 写于 2026-11-02 20:20 (Asia/Shanghai)。所有结论都有提交号或日志路径，未验证的都标为「待确认」。

---

## 0. 一句话现状

- **场景 A / B 已 100% 逐位对齐 HF**（20/20 全部 64 token 一致）
- **场景 C 长上下文：前 8 条中 7 条逐位一致**，失败机理**未查明**
- **Step 0 尚未通过**（门槛：200 条 ≥90%，且 C 未达标）
- 性能未做（参考路径 9.3 tok/s，快路径单流 33 tok/s）

---

## 1. 代码位置（全部在 `/workspace/ATOM-tritonmoe`，分支 `dsh/moe-determinism`）

### 1.1 参考路径核心：`atom/models/reference_mode.py`（约 800 行，最重要）

| 行号 | 符号 | 作用 |
|---|---|---|
| 42 | `_rotate_half` | HF 的 `cat(-x2, x1)` |
| 58 | `_moe_grouped` | MoE 前向，用 `F.grouped_mm`（与 HF 逐位一致，已验证）|
| 70 | `_unshuffle` | 逆 aiter shuffle，把 ATOM 专家权重还原成 checkpoint 布局（已验证 `un(raw) == ckpt`，0.000000）|
| 118 | `_rotary` | 直接用 HF 的 `MellumRotaryEmbedding`，按 `layer_types[li]` 取 sliding/full |
| 172 | `_reset_reference_state` | 清空 `_HIST/_KV/_WRITTEN` 与 pending/last 标志 |
| 181 | `history_after` | 按 **token 流** 判定新请求（不能用 `first_pos==0`，recompute 模式下 decode 也是 0）|
| 205 | `seq_slots` | recompute 模式下槽位就是 `arange(ctx)` |
| 224 | `reference_forward` | 主入口，由 `mellum.py` 的 `compute_logits` 调用 |

`reference_forward` 内部结构（按行号推进）：

```
~440  每层取 layer_types[li]，决定 W（滑窗）与 lt
~452  KV 缓冲分配：_own → max_position_embeddings+1（推荐），否则 ctx_lens+1（会重分配清零，已修）
~470  for s in range(bs): 每请求
~475  h = _hf_norm(x[lo:hi], input_layernorm.weight)          ← 与 HF 的 MellumRMSNorm 逐位一致
~490  q/k/v = F.linear(h, 拆分后的 qkv_proj.weight)
~495  q = _hf_norm(q, q_norm), k = _hf_norm(k, k_norm)
~500  rope: q = q*cos + _rotate_half(q)*sin
~520  KV 写入 k_flat[write_slots] = k，同时置 _WRITTEN 位图
~525  sl = ctx_slots[s]; kk = k_flat[sl].transpose(0,1); vv = ...
~545  for c0 in range(0, n_new, CH):  query 分块
~548      mask = (ki <= qi) [& ((qi-ki) < W) 当滑窗]
~560      if _hl: 逐 head 循环（调试用）
~632      if ATTN4D==1: 4-D 版 scores（**关键修复**，见 §3）
~660      否则：3-D 版 [nKV, rep*S, D]（会产生 1 ulp 差异）
~700  o = o_all → outs[lo:hi] = F.linear(o, o_proj.weight)
~710  x = x + outs
~730  h2 = _hf_norm(x, post_attention_layernorm.weight)
~735  moe = _moe_grouped(...)
~745  x = x + moe; 可选 _LDUMP 落盘
```

### 1.2 `atom/models/mellum.py`

| 行号 | 内容 |
|---|---|
| ~512 | `from atom.models.reference_mode import reference_forward` |
| ~514 | `_ref = reference_forward(self, input_ids, positions)` |
| 527 | `def compute_logits`：参考模式要把 logits **按 `cu_seqlens_q` 规约成每请求一行** |

### 1.3 `atom/model_engine/model_runner.py`

| 行号 | 内容 |
|---|---|
| 190/257 | `prev_token_ids`（deferred token 相关，注意它会跨请求带入）|
| 299 | `prepare_input_ids` |
| 313 | `total_reqs_prefill` 判定 |
| 323 | `if total_reqs_prefill > 0: return input_ids.gpu[:total_tokens_prefill]` |

### 1.4 `atom/model_engine/llm_engine.py`

- `ATOM_REFERENCE_MODE=1` 时把 tokenize 后的 prompt 写到 `/tmp/atom_reference_prompt.json`

---

## 2. 环境开关（全部在 `reference_mode.py` 内读取）

| 变量 | 值 | 说明 |
|---|---|---|
| `ATOM_REFERENCE_MODE` | 1 | 总开关，走参考路径 |
| `ATOM_REFERENCE_RECOMPUTE` | 1 | 每步重算全历史（慢但稳）|
| `ATOM_REFERENCE_OWN_CACHE` | **1（必须）** | KV 按 max_position_embeddings 预分配，避免重分配清零 |
| `ATOM_REFERENCE_ATTN4D` | **1（必须）** | 4-D scores，与 HF 的 batch 结构一致 |
| `ATOM_REFERENCE_MOE_DET` | 0 | 逐专家确定性替代；**与 HF 不等价（差 0.025）**，对比时关掉 |
| `ATOM_REFERENCE_ATTN_CHUNK` | 0 | query 分块大小，0=不分块 |
| `ATOM_REFERENCE_DUMPL` | 目录 | 逐层落盘（`layers_0.pt`）|
| `ATOM_REFERENCE_CHUNKLOG` | 1 | 打印每次 forward 的 q_len/ctx/pos/slots |
| `ATOM_REFERENCE_KVCHK` | 1 | 打印可见槽位中未写入的数量 |
| `ATOM_REFERENCE_ATTN4D_LOG` | 1 | 打印 4-D 分支形状 |
| `ATOM_REFERENCE_DEBUG / DUMP / DUMP2 / LOGSTEP` | | 其他调试 |
| `ATOM_REFERENCE_HEAD_LOOP` | 1 | 逐 KV-head 循环（慢，用于定位）|

---

## 3. 已修复的三个真实缺陷（都有提交）

| # | 缺陷 | 提交 | 证据 |
|---|---|---|---|
| 1 | **attention scores 用 3-D `[nKV, rep*S, D]`，HF 用 4-D `[1, nH, S, D]`** → 选到不同 GEMM kernel → 逐层放大并翻转 argmax | `f6d295d2` | 修后 A 20/20、1200 token 逐位一致 |
| 2 | **KV 缓冲 `_need = ctx_lens+1` 每步增长并重分配为全零**，抹掉历史 | `78da84c3` | 修后 1200 token decode 逐位一致 |
| 3 | **参考路径状态跨请求残留**（`_HIST/_KV/_WRITTEN`）| `96aa545c` | 逻辑修复，但见 §5 的教训 |

修复 1 的代码在 `reference_mode.py:632`：把 4 个 KV head 用 `repeat_interleave` 展开成 32 个 query head，`qh.reshape(1, nH, S, D)`，再 `torch.matmul`。

---

## 4. 诊断工具与数据

### 4.1 服务器启动模板

```bash
export HIP_VISIBLE_DEVICES=0
export ATOM_REFERENCE_MODE=1 ATOM_REFERENCE_RECOMPUTE=1
export ATOM_REFERENCE_ATTN4D=1 ATOM_REFERENCE_OWN_CACHE=1
export ATOM_REFERENCE_MODEL_DIR=/data/models/Mellum2.1-12B-A2.5B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup python3 -m atom.entrypoints.openai_server \
  --model /data/models/Mellum2.1-12B-A2.5B-Thinking \
  --level 0 --enforce-eager --kv_cache_dtype bf16 \
  --max-model-len 32768 --gpu-memory-utilization 0.45 \
  --host 127.0.0.1 --server-port 8001 --no-enable_prefix_caching \
  > /workspace/logs/server.log 2>&1 &
```

**注意**：改完代码必须重启服务，否则测的是旧进程（这个坑踩过多次）。

### 4.2 关键脚本

| 脚本 | 用途 |
|---|---|
| `scripts/a_4d.py` | 场景 A 一致性（20 条 × 64 token）|
| `scripts/bc_4d.py` | 场景 B / C |
| `scripts/c_clean.py` | C 的干净测量（24 token，12 条）|
| `scripts/c1200_cmp.py` / `c1200_layers.py` | 1200 token 的逐层对照 |
| `scripts/c7881_layers.py` | 7881 token 的逐块逐层对照 |
| `scripts/attn0_cmp.py` / `four_cmp.py` / `inner_cmp.py` | L0 子算子逐级对照 |
| `scripts/exp4b.py` | 三路 MoE GEMM 对照（grouped_mm vs 逐专家 vs HF）|
| `scripts/logits_cmp.py` | logits 对照（**注入未生效，待修**）|
| `scripts/tf_c11.py` | 教师强制（**方法无效，见 §5**）|

### 4.3 数据

- `/workspace/correctness/hf_scenario_a_20_eager.jsonl`、`b_20`、`c_200`
- `/workspace/correctness/hf_scenario_a_20_fresh.json`（A 的新鲜参考）
- `/workspace/logs/` 下全部日志与 `.pt` dump

---

## 5. 踩过的坑（新 agent 必读）

1. **测前必须确认服务是新代码**：多次结论被旧进程污染（C 从 0.047 变 1.000 就是这类）。
2. **文本 prompt 路径会丢尾部空白**：`tok.decode()` 拼 "\n\n" 再发 HTTP，引擎重新 tokenize 后不可靠 → 教师强制全部返回 233。**要做 token 级教师强制必须传 token id，不能传文本**。
3. **`blk_27` vs `hidden_states[28]` 不是同一语义**：前者是 final norm 前，后者是 norm 后，比较会得到巨大假差异。
4. **不要用 `first_pos == 0` 判断新请求**：recompute 模式下 decode 也从 0 开始。
5. **`MOE_DET=1` 与 HF 不等价**（累加顺序不同，差 0.025），对比 HF 时必须关。
6. **638 tok/s 是 32 路并发总吞吐**，不是单流；单流基线 25.7 tok/s，快路径 33 tok/s 更快。
7. **4-D scores 显存**：7881 token 时 `32 × 7881 × 7881 × 2B ≈ 3.7 GiB`（bf16），不是 TB。

---

## 6. 未解决 / 下一步（按优先级）

### P0：C 场景剩余的失败

**现象**：前 8 条 7 条 exact，失败条目与长度无关（10306 token 通过，4368 失败），且**机理不明**。

**下一步建议**：
1. 先修 `logits_cmp.py` 的注入（`compute_logits` 锚点不对），拿到 step0/step1 完整 logits 对比
2. 用 **token id 直传** 做教师强制（若 API 不支持，就在参考路径里直接 in-process 调用）
3. 对失败条目做 1200-token 那套逐层对照，找首个不一致层
4. 检查 `_LAST_PRED` / `_PENDING_FIRST` 是否在正确时机被消费

### P1：把 4-D 布局移植到生产 kernel

参考路径 9.3 tok/s，不可交付。生产快路径（`pa_fwd_asm` / Gluon paged attention）大概率有同样的 batch 结构问题。

### P2：性能与题目其余判据

- 32 路并发吞吐（用 `evalscope perf` 按题目命令）
- MTP 开关一致性
- 双卡一致性
- HumanEval pass@1（已测：ATOM 2.44% vs HF 2.44%，差 0.00 点 ✅）

---

## 7. 复现当前结果的最短路径

```bash
# 1. 起服务（见 §4.1）
# 2. A 场景
python3 scripts/a_4d.py            # 期望 agreement=1.0000 exact=20/20
# 3. B 场景
python3 scripts/bc_4d.py           # 期望 B 1.0000
# 4. C 场景
python3 scripts/c_clean.py         # 期望 7/8 exact，C-0011 约 0.708
```

---

## 8. 提交历史（关键几个）

```
f6d295d2  fix: 4-D attention score matmul（A 场景 20/20 的转折点）
78da84c3  fix: KV 不重分配清零
96aa545c  fix: 参考路径按请求隔离状态
b0fb60fb  diag: 教师强制实验无效的说明
797e20d7  diag: C 场景干净测量 7/8 exact
```

共享记录：`/workspace/ATOM/RESULTS.md`（约 1900 行，含全部排查过程）
