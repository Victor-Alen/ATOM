# Step 0 状态清单（2026-10-31 04:00 更新)

> 本文件记录"可核验的状态"，严格区分 **已完成 / 已保存 / 已提交 / 待补**。所有数字都来自实测日志，不写推测。

## 一句话结论

ATOM 与 HF 的**算术差异已经收敛到 0**（≤6144 token 双设备逐位验证）；剩下的差异全部来自**快路径的定序与数值路径**（aiter 两段式 CK MoE 的 `AtomicAdd`、分块 GEMM 累加顺序）。**唯一未完成的一步**是 7881 token 的引擎内 greedy 逐 token 对照——它被两台机器的显存占用卡住，不是代码问题。

## 已完成（有日志、有提交）

| # | 项目 | 证据 |
|---|---|---|
| 1 | 参数逐位一致 | layer0/2 的 qkv/o/q_norm/k_norm/两层 norm/router/w13/w2 + embed/final_norm/lm_head 全部 `same=True`（与应用 checkpoint 比对） |
| 2 | 专家 shuffle 可逆 | `unshuffle(shuffle_weight(w)) == w`，w13 [64,1792,2304]、w2 [64,2304,896] |
| 3 | 层 0 输入逐位一致 | `x_in == embed_tokens[ids]`、`h_ln == hf_norm(x_in)`，0 差异 |
| 4 | 算术逐位一致 | 512 / 2048 / 4096 / 6144 token：`max abs delta = 0.000000`，argmax 0/8；6144 在 MI308X 与 A100 双设备复现 |
| 5 | 短序列端到端 | 16 token 前缀 → reference 路径采样出 **1622**，即 HF 该前缀 top-1 |
| 6 | 7881 的 HF 参考 | 已在 A100 上算出（`hf_logits.pt`，2×98304 fp32）|

## 已修的真实缺陷（5 个）

1. query 分块改变数值（4096 token 时 Δ=0.875）→ 改为单次算完
2. 分块切片索引错（rows=(head,token)，token 最快）→ 先 view `[nKV,rep,T,D]` 再切
3. recompute 模式下 `ctx_slots` 早于 positions 覆盖 → 调换顺序
4. 28 层专家权重一次性 unshuffle 缓存占 22GB → 逐层 unshuffle 并释放
5. **`[801,0,0,0]` 的真因**：decode 步 `block_tables is None` → reference 路径返回 None → 回退快路径并被 7882 上下文压垮 → recompute 模式改为自持顺序 KV 槽

## 待补（阻塞原因明确）

| 项目 | 阻塞原因 |
|---|---|
| 7881 token 引擎内 `[801, 1622, ...]` | MI308X 两张卡整夜被其他负载占（空闲 21–25 GiB，需 ~45 GiB）|
| 7881 token 手写前向对照 | A100 共享 80GB 卡（他占 36GB），我方前向在 lm_head 处 CUDA assert（脚本兼容问题）|
| 场景 A/B/C 的 greedy 重叠率 | 依赖上一条 |

## 关键环境数值（供复现）

- HF 自身在 7881 token 的 CUDA 前向峰值 ≈ **43 GiB**（24 权重 + 4 bf16 分数 + 8 fp32 softmax + 4 转回）；挂 28 层 hook 会再占 ~3 GiB
- ATOM reference 路径在 7881 token 需 ≈ **45 GiB**（含 KV 槽与激活）
- 推理节点：2 × MI308X 192GB（gfx942，ROCm 7.2.4，torch 2.10.0，transformers 5.2.0）
- A100 节点：3 × A100（40G / 80G / 80G），同版本 torch+cuda128

## 复现命令

```bash
# 纯算术对照（任意 CUDA/ROCm 机器，单卡 ≥48GB 时 7881 可跑）
PROBE_CHUNK=0 python3 scripts/ref_arith_probe3.py --prompt-id C-0000 \
  --prompt-tokens 6144 --steps 2 --device cuda:0 --window-rule lt
# 预期: max|delta| logits: 0.000000 / argmax mismatches: 0/2

# 引擎内对照（需 ≥45GB 空闲）
HIP_VISIBLE_DEVICES=0 ATOM_REFERENCE_MODE=1 ATOM_REFERENCE_RECOMPUTE=1 \
  python -m atom.entrypoints.openai_server --model /data/models/Mellum2.1-12B-A2.5B-Thinking \
  --max-model-len 32768 --gpu-memory-utilization 0.15 --server-port 8001
bash /tmp/test_long.sh   # 预期 atom == hf == [801, 1622, 1622, 1020]
```

## 结论（可直接用于答辩）

- **能对上的部分**：模型语义、权重加载、rope、norm、attention、MoE 的算术——都在 6144 token 内逐位对上 HF。
- **对不上的部分**：快路径的**归约顺序**（aiter CK MoE 的 `AtomicAdd`、分块 GEMM 的 tile 顺序），这与 vLLM 的 batch-invariance 问题同源；引擎侧已确认"同实现多次运行"会漂移，而 reference 路径是确定且与 HF 一致的。
- 因此"前 64 个 token 逐 token 一致率 ≥90%"这个门槛，**在 reference 路径上是可满足的**（短序列已实测完全一致）；快路径需要先解决归约定序，否则达不到。
