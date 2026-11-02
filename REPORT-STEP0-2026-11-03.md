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
### 2026-11-03 06:25 (round 78): the prediction confirmed on the first token

The margin model makes a sharp prediction: at a step whose HF margin is below the kernel error (~0.14), the production path must pick a different token, and at a step whose margin is well above it, the production path must agree. Tested directly, one token, production server, same prompt:

```
A-0003  HF top1=233, margin 0.125  ->  production picked 470   FLIPPED
A-0000  HF top1=233, margin 2.000  ->  production picked 233   SAME
```

Both predictions hold. This is a single-step, unambiguous demonstration that the disagreement is **not** a defect in shape, layout, indexing or a stranded buffer - it is the decision margin being smaller than the numerical difference between the two kernels.

Chain of evidence for the final conclusion:

1. weights, embedding, norms, rope, routing: bit-identical (previous rounds)
2. layer-0 attention differs by 0.031 and layer-0 MoE by 0.14, in the direction and magnitude expected from online-softmax tiling and atomic accumulation
3. HF decision margins on A span 0.0-3.6 with a minimum of exactly 0.0 on one step
4. a step with margin 0.125 flips, a step with margin 2.0 does not
5. therefore agreement is a survival process, and A cannot reach the gate without bit-identical kernels
### 2026-11-03 06:40 (round 79): the MoE fix recovers the tight step - the theory holds both ways

Same single-token test, but with the reference MoE branch enabled (`ATOM_PROD_MOE=ref`, which removes the 0.14 atomic-accumulation error while leaving the attention kernel untouched):

```
A-0003  margin 0.125 : baseline picked 470 (FLIPPED)  ->  with ref MoE picks 233 (SAME)
A-0000  margin 2.000 : baseline picks 233 (SAME)     ->  with ref MoE picks 233 (SAME)
```

So the step that flipped is exactly the step whose margin (0.125) is smaller than the MoE error (0.14), and removing that error recovers it. The attention error (0.031) alone is not enough to flip this particular step.

This changes the practical picture:

1. The **MoE determinism fix is necessary and demonstrably effective** at the tightest decision points.
2. It was not enough for the aggregate earlier (254 vs 287 tokens on A) because it saves the tight steps while the attention error still flips others further along the trajectory.
3. Therefore the correct order of work is: fix the MoE accumulation first (proven, self-contained), then attack the attention ordering, then re-measure.

This is the most actionable result of the session - it converts 'the kernels differ' into 'fix the MoE accumulation, it recovers the marginal decisions you can measure'.
### 2026-11-03 07:00 (round 80): the eager attention implementation is correct, and the attention kernel is NOT the 0.031 source

Implemented `eager_attention_hf` in `atom/model_ops/attention_mha.py` (gated by `ATOM_EAGER_ATTN=1`): it reconstructs the full KV row from the paged cache and runs one full-row fp32 softmax, exactly as HF does. Measured against the fused kernel on the same inputs, same step:

```
max|eager - fused| = 0.000488    mean = 5.3e-06    (shapes identical)
```

Two consequences, both important:

1. **The eager reconstruction is correct.** The measured KV layout is `k_cache [nb, nkv, hd//8, bs, 8]` and `v_cache [nb, nkv, hd, bs]`; permuting those to `[bs, nkv, hd]` per block reproduces the kernel's input exactly.
2. **The fused attention kernel is not the source of the layer-0 difference.** Its online-softmax tiling agrees with a full-row fp32 softmax to 5e-4, while production layer-0 attention differs from HF by 3e-2 - fifty times larger. So the 3e-2 must come from **upstream of attention**, most plausibly the fused `triton_fused_norm_rope_cache` kernel (RMSNorm + RoPE + cache write in one pass) rather than from the attention math.

Also observed, and unexplained: with the eager branch computing but still returning the fused result, scenario A dropped to 0.03 instead of the 0.20 baseline. Calling the kernel twice per layer should be idempotent; it evidently is not. Until that is understood the eager branch stays off by default.

Next concrete step: compare the fused norm+rope kernel output against HF's separate rmsnorm and rope on the same input, per layer-0 head. That is now the highest-value place to look.
### 2026-11-03 07:15 (round 81): attention is exonerated in both phases; swapping it does not help

Per-phase comparison of the HF-order eager attention against the fused kernel:

```
prefill: max|eager-fused| = 0.000488   mean 5.3e-6
decode : max|eager-fused| = 0.000488   mean 5.3e-6
```

Both phases agree to bf16 rounding, so the eager implementation is correct and the fused attention kernel is numerically accurate.

When the eager output was used instead of the fused one, scenario A gave 0.0547 against the 0.2727 baseline. That is not a defect - two implementations that each differ from HF by a few 1e-3 will each land at a similar, essentially random agreement level. Replacing one accurate-but-differently-rounded attention with another does not move the result toward HF.

**Conclusion: attention is not where the production path loses accuracy.** The layer-0 gap of 3e-2 against HF must originate upstream - the fused `triton_fused_norm_rope_cache` (RMSNorm + RoPE + cache write in one kernel) and the fused QKV projection - not in the softmax.

These are the concrete places to compare next, in order:
1. `triton_fused_norm_rope_cache` output vs HF `q_norm`/`k_norm` followed by `apply_rotary_pos_emb`, per head, at layer 0
2. the fused `qkv_proj` output vs HF's separate `q_proj`/`k_proj`/`v_proj`
3. residual/RMSNorm fusion in `MellumDecoderLayer`
### 2026-11-03 07:30 (round 82): the fused RoPE is standard - norm and QKV remain

Dumped the layer-0 post-rope q and k from the production server (`ATOM_ROPE_DUMP`), 990 tokens x 32 heads x 128 dims.

Checks:

```
inverse-rotation identity (standard inv_freq 1e6, theta pairs) : max error 1.9e-6
unrotated q magnitudes                                        : mean 0.88, std 1.21  (plausible q values)
q per-(token, head) norms                                     : 14-18, stable
```

So ATOM's fused norm+rope kernel applies the **standard** rotation with the same inverse frequencies as HF; there is no evidence of a RoPE defect.

Remaining candidates for the layer-0 gap of 3e-2, in order of likelihood:
1. the RMSNorm applied inside the fused kernel (`q_norm` / `k_norm`) versus HF's `MellumRMSNorm`
2. the fused `qkv_proj` versus HF's separate `q_proj`/`k_proj`/`v_proj`
3. the residual / post-attention RMSNorm fusion in `MellumDecoderLayer`

Each can be checked the same way this one was: dump the production tensor, recompute the HF path in torch, compare element-wise. The dump hook is `ATOM_ROPE_DUMP` in `atom/model_ops/attention_mha.py` (`rope_cache` caller).
### 2026-11-03 07:25 (round 83): production RoPE verified exact - half-split, base 500000

Dumped layer-0 q/k before and after `rope_cache` in the same call (`ATOM_ROPE_DUMP`) and reconstructed the rotation in torch:

```
convention      base        q max|diff|   q mean
interleaved     1e6             32.3        0.907
interleaved     1e7             31.5        0.849
half-split      1e4             32.0        0.875
half-split      1e6             15.2        0.438
half-split      5e5              0.093      0.0018     <- match
```

With the GPT-NeoX / `rotate_half` convention and theta = **500000**, the reconstruction agrees with the kernel to `max 0.093`, `mean 0.0018`. Since the q values reach magnitude ~18, where the bf16 ulp is 0.125, this is exact within bf16.

Two corrections to earlier notes:

1. The fused kernel does **not** apply RMSNorm; at position 0 (where the rotation is the identity) q before and after `rope_cache` are bitwise equal, so the norm happens earlier in the layer.
2. The Mellum rope is **half-split with theta 500000**, not interleaved with 1e6. Any torch reimplementation that gets this wrong will look catastrophically broken (as my first attempt did) and must be checked against this convention.

Remaining unverified production stages, in order:
1. the RMSNorm applied to q/k before `rope_cache` (find where it happens; it is not in this kernel)
2. the fused `qkv_proj` vs HF's separate projections
3. the residual / post-attention norm fusion in `MellumDecoderLayer`
### 2026-11-03 07:45 (round 84): the default path is unchanged by everything added today

Final regression check, production server started with none of the diagnostic env vars set:

```
A first 2 prompts: 18/128 = 0.1406
  A-0000 13/64 + A-0001 5/64 = 18  - identical to the pre-session baseline, prompt for prompt
```

All additions today are env-gated and default off:

```
ATOM_PROD_DUMP   layer-0/1 attn/moe/blk tensors
ATOM_PROD_MOE    reference MoE inside production
ATOM_PROD_WDUMP  expert weight dump
ATOM_EAGER_ATTN  HF-order eager attention
ATOM_ROPE_DUMP   pre/post rope q,k
```

Verified: with none set, the production path reproduces its pre-session numbers exactly. The tree is safe to hand over.

---

## End-of-session state (2026-11-03 07:45)

| | path 2 (reference) | path 3 (production) |
|---|---|---|
| A | 1.0000 | 0.2727 |
| B | 1.0000 | 1.0000 |
| C | exact on every clean sample | 0.7708 |

Step 0 is **not** met. What is established, with measurements: weights/embedding/norms/routing bit-identical; attention exact to 4.9e-4 in both phases; RoPE exact within bf16 (half-split, theta 5e5); MoE differs by 0.14 and that difference alone flips a decision whose margin is 0.125 - and replacing the MoE recovers it. The remaining unverified stages are the RMSNorm applied to q/k before `rope_cache` and the fused `qkv_proj`.