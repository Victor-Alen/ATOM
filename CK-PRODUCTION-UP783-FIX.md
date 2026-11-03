# 生产 CK Up783 修复与部署验证（2026-11-03）

目标点已在实际生产 CK 中修正，并部署到8003生产候选服务。**Step 0仍未通过**，不能把本次局部修复等同于整模型通过。

## 生产现场验证

- L0 / prompt token0 / expert49 / Up783：旧值`0.034423828125`，新CK与HF均为`0.0341796875`。
- 原先修正的L0 / token0 / expert55 / Gate407保持`-0.05517578125`，与HF一致。
- **L0 token0的Gate、Up各7,168个值，全量与HF一致**。相同现场输入的服务原始GEMM输出与独立CK重放，各7,096,320值全量一致。
- L0 input norm、attention、post norm各2,280,960值仍与官方HF一致，实际新norm调用57次。
- 最新CK stage1全量仍有Gate **1,387**、Up **1,394** 个不同值（各7,096,320）。目前stage1最早首差：**token1 / expert17 / Up275**，CK`9.107589721679688e-05`、HF`9.059906005859375e-05`。
- 最新L0 MoE最终输出仍有6,088 / 2,280,960不同值，max`0.0009765625`、mean`3.8310886907311215e-07`。

## 实际修改的生产路径

1. 从当前安装hipBLASLt查询BF16 gate/up GEMM（N1792/K2304）的dispatch配置，按专家实际token数选择MI16/MI32的K分组；查询覆盖M1..32768，现场捕获的51个活跃专家kernel名全部吻合。该表保存配置，不保存参考输出或目标坐标。
2. GPU整数计数生成每expert的分组标记，沿扩展sorting header传给CK；Gate和Up共享规则。
3. CK V3 stage1按标记选择register/lane的K分组，并移除该变体中旧Up FP64 chunk merge。**当前二进制全部5个V3 stage1实例均已重新编译**，不是只修A-0000用到的实例。V1 decode不套用该分组修复，且不生成这份prefill元数据。
4. stage2使用原sorting header；12个stage2对象与冻结基线逐字节一致，未改变stage2 GEMM或逐slot定序reduce。
5. 新元数据做了GPU架构、hipBLASLt及CK二进制SHA256校验，避免误配旧库。模型MoE GEMM仍由CK执行，没有改成PyTorch逐专家循环。

StaggerU、LocalSplitU完整算术尚未落实到CK；不能把独立Triton探针“Gate1/Up4值不同”当作当前CK的误差计数。下一个首差需要继续隔离e17的局部FP32累加与LocalSplitU=4合并；具体首次不同的MFMA/合并步骤尚未确定。Stage2固定HF stage1时残余173个最终值仍未修复。

## 版本与运行状态

- ATOM运行代码：`9082ed20`（基础接入`b8402cad`）。分支`dsh/mellum-attention-kv-isolation`。
- CK：`5082d415d`（分组接入`2a17cee0c`），分支`mellum/hf-profile-korder`，目录`/workspace/ck-mellum-hf-profile`。
- AITER源码：`8f3451375`；JIT目录`/workspace/ck-mellum-hf-profile-jit`。
- MoE二进制SHA256：`85e900a3b6928d9b88cb9360de43c740fed29b52469ae2d135fe41ed1b4c89c8`。
- 活跃服务：127.0.0.1:8003，PID **856843**；已用真实生成及GPU显存验证，不只检查health。
- 启动器：`/workspace/ATOM-tritonmoe/scripts/start_mellum_ck_profile.sh`。所有历史冻结版本保留；未推送。
- level0 / enforce-eager / BF16无量化 / BF16 KV / maxlen32768 / util0.45 / 禁prefix cache / 单客户端。保留此前attention、norm、RoPE、QKV候选开关；reference mode关闭。

## 双卡确定性与元数据验证

固定同一生产MoE输入、权重和routing，两GPU各40次重复全相同，双卡结果相同；只证明这个固定fixture，不外推为全模型确定性。元数据验证覆盖1、990、2240 token，包含padding计数边界。Python语法、启动器语法、git diff --check通过；ruff未安装，未运行ruff。

## 两例性能（各一次，64输出token）

同一GPU、相同请求、eager、禁prefix cache。指标为`64 / 请求总耗时`，包含prefill和服务开销，**不是纯decode吞吐或严格性能回归结论**。

| 请求 | 旧服务 tok/s | 新服务 tok/s | 速度变化 | 旧/新匹配token |
|---|---:|---:|---:|---:|
| A-0000（990 prompt tokens） | 24.632 | 23.897 | -2.98% | 5/64 → 3/64 |
| A-0001（2240 prompt tokens） | 24.118 | 23.424 | -2.88% | 64/64 → 64/64 |

初版候选曾因decode重复生成用不到的元数据而慢约8%，最终版本已移除，以上为最终版本结果。

## A、C各10请求，前64 token现场复测

HF均使用官方`model.generate()`：greedy、use_cache=True、max_new_tokens64、eos_token_id=None、pad_token_id0。A10现场重算全部复现历史参考；C10从原始dataset重新分词与生成，**未使用旧C输出**。C-0000、C-0001重复生成均相同。用实际ATOM tokenizer函数逐条核对，20个请求的输入token IDs均与HF完全相同。

| 场景 | 匹配token位置 / 总位置 | 一致率 | 整条64 token相同的请求 | ≥0.90 |
|---|---:|---:|---:|---|
| A10×64 | 131/640 | 20.4688% | 1/10 | 未通过 |
| C10×64 | 320/640 | 50.0000% | 4/10 | 未通过 |

这里统计自由greedy生成的逐位置token ID是否相同，同时单列整条64-token完全相同的请求数。局部数值对齐没有使本轮Step0通过。

新HF参考SHA256：A `222d4dc6ed6848eaeaffb88da7fb7d54276d99b11cf339a2a305f17ed17e9efe`；C `8fa9ff1797873186bf044e37546d957ccb8646b4ed74d16c7c694839d2a05cb8`。

## 证据与复现

根目录`/workspace/logs/step0_continuation/ck_hf_profile_prod/`：

- `final_summary.json`、`runtime_manifest.json`：汇总、实际配置与二进制。
- `service_final/actual_kernel_validation.json`及`atom/step_0.pt`：真实服务内原始Gate/Up、旧首差与新首差。
- `fixed_validation_gpu0.json`、`fixed_validation_gpu1.json`：40次与跨GPU验证。
- `baseline_pair.json`、`candidate_final_pair.json`：两例性能及完整生成token IDs。
- `fresh_hf/manifest.json`、`A10_64.jsonl`、`C10_64.jsonl`：新HF输出与复现检查。
- `A10_64/`、`C10_64/`：逐请求结果、原始响应、summary；对应log保留每条首个生成分叉位置。
- `build_all_v3.log`、`heuristic32768.tsv`：全部5个V3实例编译与配置查询证据。

生成与验证脚本随本次提交入库。本轮不是三端口同批验收，也没有完成200请求或MTP验证。
