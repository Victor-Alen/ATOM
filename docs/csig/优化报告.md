# Mellum / ATOM 推理优化报告

蒋天颢 · 腾讯 CSIG 云一计算加速题目 · 2026 年 11 月 6 日

## 1. 交付结论与实验口径

**单卡 BF16、MTP 关闭，恰好三个优化点：MoE 专家计算、Attention 计算、分段编译与 Graph 执行。** 完整测试核心版本为 `0453ef43`。A 吞吐三轮中位数 **699.168 token/s/GPU，比题目基线高 9.54%**；B 单流 TPOT p50 为 **9.240 ms**；C 两项 p99 仍高于基线。后续交付提交只补材料与验证入口，模型源码以哈希绑定；不将公开集结果写成隐藏集验收。[E1–E3]

| 指标 | baseline.json | 三轮完整实测 | 三轮中位数与判断 |
|---|---|---|---|
| A tok/s/GPU ↑ | 638.2827 | 699.0880 / 699.1678 / 702.3639 | 699.1678；提升 9.54% |
| B TPOT p50，ms ↓ | 未提供 | 9.2401 / 9.1946 / 9.2440 | 9.2401；不推算基线加速比 |
| C TPOT p99，ms ↓ | 41.2650 | 64.4984 / 64.5116 / 64.5111 | 64.5111；为基线 1.56 倍 |
| C TTFT p99，ms ↓ | 993.5490 | 3948.2632 / 3964.4639 / 3940.2434 | 3948.2632；为基线 3.97 倍 |

**基线来源。** `/data/datasets/baseline.json` 记录单卡、eager、compile level 0、BF16 KV、MTP off、SWA on。638.28 是其中 A 的输出吞吐，不是 B 单流速度，也不是本次另跑 HF 得到的性能。附近 `humaneval_bf16.json` 的3/164被 baseline 标为聊天模板/思考模式不一致的无效协议，不作精度对照。[E1]

**负载与计时。** 冻结容器、一张 MI308X，PCI `0000:22:00.0`；A512、并发32；B50、并发1、输入1024；C200、开放环泊松率0.30请求/秒、单请求超时180秒。均流式输出256 token、greedy、忽略 EOS，种子20261105。每场景先算每轮分位数，再对三轮指标取中位数，不混合请求重算p99。共2286个主测试请求全部成功且均输出256 token；源码与依赖不变，正式计时不开 profiler。[E2]

**测量边界。** 服务只选择HIP1；同机另一卡另有独立服务，不称为整机空闲测量。EvalScope 1.14.1不接受文件路径作dataset名，A用`line_by_line`原样读取，不截断或补齐；512条中5条实际长4097–4113。旧版过滤补充不替换本版结果。同进程正确性检查及预热后计时，持久编译缓存已存在，不称为冷启动性能。

## 2. 完整正确性门槛

C200前64 token **12800/12800一致**；独立16384-token输入 **64/64一致**；HumanEval **148/164=90.24%**，同协议HF也是148/164，差距 **0个百分点**，全部164条生成序列一致。HumanEval采用聊天模板、`enable_thinking=False`、greedy、EOS停止、最大2048 token及固定官方checker，双方均有3条达到长度上限。[E3]

C200对照固定哈希的HF eager参考，HumanEval也沿用固定同协议参考。另新跑HF eager，先核对两个历史锚点，再生成3条不同内容、恰好16K的参考；本版串行及并发3合计6请求 **384/384 token一致**。这是3条独立输入的补充精度检查，不是性能数据或隐藏集验收。

<!-- pagebreak -->

## 3. Profiling 证据与三个优化点的选择

选择顺序是“正确性兼容路径 → 工具时间线 → 单点替换 → 整模型验证”。逐请求GEMM、显式Attention和专家循环保留了大量启动、权重搬运与窗口外计算。三个点分别针对专家计算、Attention、重复执行提交；Norm/Linear/RoPE的基础精度适配属于共同Step 0，不包装成第四个点。

**历史工具证据。** 中间版本A32 prefill的torch profiler原始trace有32159个GPU kernel事件，累计9.216秒；MoE占34.8%、GEMM占32.4%、其他32.8%。GEMM也包含Attention，累计事件时间可能跨stream重叠，不是请求墙钟。该采样用于选择方向，不冒充最终版本占比。[E5]

**最终同版输出。** `final_delivery_profile_20261105T224908Z` 使用完整打包依赖，预热后采集短输入输出64、16K输入输出8、A32各输出8三个窗口，GPU事件累计分别为 463.44 / 1745.25 / 10076.60 ms。16K窗口共5692事件，其中MoE 468.28ms、softmax 396.75ms、GEMM 302.27ms、QK核 283.43ms。名称分类与原始trace均保留；窗口包含少量decode，不称为纯prefill，也不把诊断耗时当作正式p99。[E5]

### 点 1：MoE 专家计算

64专家、top-8、hidden2304、专家中间维896。恢复原生CK两阶段批量专家计算，复用路由与缓冲，融合BF16路由，专家写独立slot后有序归并，减少重复路由、碎片GEMM与搬运；按已验证形状选择stage1/stage2参数。两条prefill stream属于同一专家计算实现。

**原核为什么不能直接用。** 中间BF16舍入、K维累加和top-k加权/输出归并顺序不同，局部误差会改变后续greedy分支。修复先定位同输入中间张量，再保留数值边界与有序输出，不以“权重相同”或“局部接近”代替整模型验收。入口为 `mellum_native_decode_moe.py`、`mellum_native_prefill_moe.py` 及打包AITER/CK修改源。[E4]

### 点 2：Attention 计算

21层1024滑窗、7层全注意力。decode使用修正的AITER/Gluon；prefill对滑窗仅计算有效QK/softmax/PV，全局QK跳过未来块，softmax按逻辑lane缓存中间值。保留HF的BF16舍入与归约顺序。参数来自冻结hipBLASLt的1–16384长度形状查询，不按提示内容或答案选择；未知形状、布局、dtype回退。

**逐核反例。** QK统一KW8在7169长度出现8177个BF16不同值，改为MI/depth/row-tile分派；PV主循环KW8、尾部KW4，修复4095长度4个尾行差异。softmax保持ATen 512-lane归约树、行对齐与指数下溢处理。窗口外概率不写入时，配对PV每次读取都施加掩码，不能交给无掩码dense fallback。[E6]

**固定形状收益。** 16K窗口PV 12.19→2.65ms，窗口softmax 30.28→5.84ms；全局QK 52.801→26.510ms。各核有边界长度与40次重复记录，之后再走模型门槛。三角QK调度另测26.521→25.186ms，只减少空任务，不能说算术量再次减半。微基准不是C p99收益比例。[E6]

<!-- pagebreak -->

### 点 3：分段编译与 Graph 执行

启用 `ATOM_MELLUM_COMPILE_PIECES=1` 编译稳定的小计算段，再用ATOM decode graph捕获批次1/2/4/8/16/32。把动态请求元数据与可重放计算分开，维持地址、偏移、缓冲生命周期，减少每token的Python提交与小算子启动。`ATOM_MELLUM_GRAPH_PIECES=0`，不叠加另一套分段graph；命令 `--level 0` 与显式片段编译是两个层次。

关闭点3时强制eager并关闭片段编译。稳定元数据与数值形状约束在eager对照也保留，消融比较编译与重放。启动/图捕获日志与预热后的稳定测量分别保留，图执行也重新验证输出；实际开关与展开环境写入manifest，不仅凭配置等级判断路径。[E4]

## 4. 累加消融与预估收益对照

以下四行均为最终核心版本、同卡、同模型/依赖、同种子与请求协议，**A64/B8/C12，每配置1轮，输出256 token**；每行先通过A8+C4+严格16K的832-token检查。C p99样本少，只作方向判断；第1页三轮完整结果单独报告，不把小测当全量。[E7]

| 配置 | 卡数 | A tokens/s/GPU ↑ | B TPOT p50 ms ↓ | C TPOT p99 / TTFT p99 ms ↓ |
|---|---|---|---|---|
| Step 0 | 1 | 63.0774 | 49.0203 | 286.2389 / 21026.3087 |
| + 点 1 | 1 | 105.0215 | 46.0112 | 243.7530 / 15808.3974 |
| + 点 1 + 2 | 1 | 447.9365 | 42.7363 | 84.2889 / 2041.3974 |
| + 点 1 + 2 + 3 | 1 | 732.0104 | 9.2048 | 37.4471 / 1988.4069 |

**预估登记。** 11月6日01:34、首次补充消融前登记，属于复盘估算，不冒充D1预先预测。下表实测按最终同版消融的相邻行计算，不相乘独立微基准倍率。[E8]

| 点 | 预估及依据 | 相邻配置实测 | 解释 |
|---|---|---|---|
| 1：MoE | 历史采样占34.8%；假设减半，理想GPU时间少17.4%、吞吐升21.1% | A提升66.50%；B降低6.14% | 历史采样不同于Step 0；不是同条件上界，A/B收益不同 |
| 2：Attention | 两个16K窗口核约省33.98ms/层，21层约714ms；未含全局层/排队 | A提升326.52%；B降低7.12% | 本点包含完整Attention路径，局部微基准不是端到端预测 |
| 3：编译与图 | 假设B TPOT降低5%–20%；缺CPU占比，低置信度 | A提升63.42%；B降低78.46% | 预估明显偏低；不能将全部差额都归因于CPU |

**边界。** Step 0是本项目可验证的兼容实现，不等同于出题方baseline。原约50 token/s的现象不能归咎于一个“改坏的算子”：逐请求形状和显式计算保住精度，也压低并行度。消融区分各段收益，完整验收决定默认版本；最终C仍未达到题目基线。

<!-- pagebreak -->

## 5. 取舍、已知限制与后续方向

**MTP未采用。** 短输入小测收益不能代替长上下文、混批、缓存、图捕获及on/off逐token一致性验收；遵照交付范围默认关闭，不计入三个点。**量化、双卡未采用：** 前者需重新验证精度，后者需完整通信与每卡归一化收益证据，本次只交单卡BF16。

**前缀缓存与分块prefill未纳入。** 题目允许这些方向，但会改变实际prefill范围、矩阵形状或专家分组，尚缺新的正确性与完整C收益证据；当前保留关闭设置，不并入已选三点。**原始融合prefill未采用：** 全部层、仅全局层、仅滑窗层的C200一致率分别46.37%、55.30%、45.41%，均未达90%。保留分拆QK/softmax/PV和条件回退，不能宣称所有原始融合核都已恢复。[E6]

**收益反例。** 新测全局softmax/PV融合虽固定输入一致，16K却从51.16增至82.04ms，故放弃；减少物理warp也在若干长行退化。C剩余问题继续从最终时间线、长prefill与decode重叠定位，不更改负载、替换HF参考或选最好一轮掩盖差距。

## 6. 代码仓库、开关与复现

在题目冻结Linux容器、模型已位于 `/data/models/Mellum2.1-12B-A2.5B-Thinking` 时，仓库根目录执行：

```bash
bash scripts/start_server.sh
/opt/venv/bin/python3 scripts/validate_public.py
bash scripts/benchmark.sh --repeat 3
```

脚本校验源码及依赖SHA256、自动解包运行库、加载预热并完成真实8-token生成后才打印READY。默认8000端口、三点全开、MTP off；可指定 `--gpu`、`--port`、`--host`。本次HIP设备1、ROCR设备0,1，但仅使用一张物理卡，不停止其他服务。[E4]

| 点 | 独立开关 | 累加配置 |
|---|---|---|
| 1：MoE | `--moe-opt 0/1` 或 `OPT_MOE=0/1` | `--ablation point1` |
| 2：Attention | `--attention-opt 0/1` 或 `OPT_ATTENTION=0/1` | `--ablation point12` |
| 3：编译与图 | `--compile-graph-opt 0/1` 或 `OPT_COMPILE_GRAPH=0/1` | `--ablation point123`（默认） |

`--ablation step0`关闭三点；切换前用 `python3 scripts/stop_server.py --port 8000` 停止本仓库服务。`--screen`仅用于小测。ATOM与原生AITER/CK修改源、生成实例、构建记录、二进制哈希随仓库交付；默认使用冻结环境中已验证二进制，**未声称全新环境从零重建通过**。完整Git历史保留、不squash；一页AI说明、D1–D7及补充日志另交。

## 7. 证据索引

交付目录 `evidence/` 保存原始记录；仓库 `results/` 有同名汇总与归档。版本、运行路径、逐文件哈希见完整证据索引。

**E1** `final-current/baseline.json`（SHA256前缀14674cbbc641）；**E2** `final-current/model/performance/`，含9组客户端SQLite、summary、percentile、命令及逐请求审计；**E3** `final-current/model/correctness/`、`additional_long_context/`及HF参考生成记录。

**E4** 仓库 `README-CSIG.md`、`scripts/`、`deployment/runtime-manifest.json`；**E5** `final-current/profiling/` 原始trace与digest，历史 `historical/profiling_digest.json`；**E6** `full-current/operators/`、`late-control/`、`final-current/experiments/`；**E7** `final-current/ablation/` 的四行原始记录与启动manifest；**E8** `optimization-estimates.md`。最终归档为 `final-acceptance-evidence-20261106.tar.gz`，校验值见同名manifest。
