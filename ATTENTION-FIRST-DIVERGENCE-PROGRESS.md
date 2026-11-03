# Mellum 首个分叉点排查续测（2026-11-03）

本轮沿官方 HF `model.generate()` 的 L0 完整现场继续定位。Step 0 仍未通过；本轮独立探针结果不替代生产端到端成绩。

## 基准与当前生产成绩

- 正式 HF generate 已完整复现 A-0000 历史前64 token。此前手写 cached greedy 与官方 generate 不同，不得作为验收参考。
- ATOM `68fc7f1a` 实际服务调用新 norm 57次，L0 input norm、attention、post norm全量与官方HF一致。
- 同轮生产 A-0000：**5/64**，首输出分叉索引3。未跑最新A10/C10，不能声称达到0.9。
- 正式A参考SHA256：`1f1bf047d422423076e102b800811723097340878adda3d275c3ebd6b6924fbd`。

## 固定 Routing 的关键隔离

冻结实际 hidden states、preshuffle 权重、CK stage1 与 stage2 配置后，重放逐值复现生产 stage1、stage2逐slot及最终输出。替换为HF routing后，按expert对齐的全部阶段及最终输出仍相同。409个slot顺序变化对应相同expert集合及等权重，本例不是选错专家。

两种routing各40次稳定；固定fixture跨两GPU全量一致。这里只证明该固定MoE输入的确定性，未证明整模型跨run/GPU确定性。

HF全批MoE重建先验真：最终2,280,960值与官方现场完全相同；checkpoint权重shuffle后与生产权重完全相同。

| 固定输入比较 | 不同值 / 总值 |
|---|---:|
| 原CK Gate GEMM | 1,576 / 7,096,320 |
| 原CK Up GEMM | 1,367 / 7,096,320 |
| 相同Gate输入的SiLU | 0 / 7,096,320 |
| 相同SiLU/Up输入的乘法 | 0 / 7,096,320 |
| 原CK最终MoE | 7,152 / 2,280,960 |
| HF stage1 → 原CK stage2逐slot | 2,079 / 18,247,680 |
| HF stage1 → 原CK stage2最终 | 173 / 2,280,960 |
| HF逐slot → 当前定序reduce最终 | 0 / 2,280,960 |

## 首差的实际汇编与 CK 实验

原首差：L0/token0/slot0/expert55/Gate feature407，CK `-0.055419921875`，HF `-0.05517578125`。FP64点积仅用于确认BF16舍入边界，未把生产MoE改为FP64。

实际HF expert55使用MT192x128x64、MI32x32x8；CK使用256线程/128x128x64、MI16x16x16。仅改变MFMA16/32或转置操作数无效。解包实际HF code object并检查LDS读及MFMA顺序后，确认需同时对齐：

1. MI32每16个K元素按`0..3,8..11,4..7,12..15`分组；MI16 KPack8使用另一种分组，不能统一套用。
2. 按HF token tile、DepthU及StaggerU旋转K块。
3. 小M实际HF配置含LocalSplitU=2或4，需要独立局部累加后按HF顺序合并。

独立Triton探针（不是生产替换）：11个非LocalSplitU专家，Gate与Up各5,575,808个BF16值全量0差异；其中expert55、7、23、49等均完整张量精确。扩展到全部51个活跃专家（含40个LocalSplitU配置）后，Gate剩**1/7,096,320**，Up剩**4/7,096,320**。仍有5值，不能称全部精确。

该探针当前最早剩余差异是 **token1 / expert17 / Up feature275**（拼接gate/up索引1171）：探针`9.107589721679688e-05`，HF `9.059906005859375e-05`。另外4值位于e32 Gate、e40/e52/e53 Up。实际LocalSplitU4汇编确认FP32合并为`((p0+p1)+p2)+p3`；剩余差异尚未定位到具体局部MFMA步骤。

实际CK独立分支 **`mellum/hf-korder`，提交`80f4da570`**：只重编译当前被选的stage1 V3对象，交换KPack寄存器与KLane维，并在该实验中关闭旧Up FP64 chunk merge；stage2对象不变。固定现场重放确认原Gate首标量修正。全批Gate剩1,512值、Up剩1,483值，gated剩2,640值；说明此局部CK实验不能作为通用修复或默认生产版本。

CK实验的Gate最早差异已移至token4/e6/feature326，但整体MoE最早仍为token0/e49/Up783，必须优先处理后者。独立探针已证明e49全量可对齐，实际CK尚未实现按HF配置选择完整K分组、stagger与LocalSplitU。

## 版本、运行范围与证据

- ATOM分支：`dsh/mellum-attention-kv-isolation`；生产数值基线`68fc7f1a`。
- AITER：`/workspace/aiter-mellum-norm-hf`，`8f3451375`；生产CK冻结`/workspace/ck-mellum-up-fp64`，`14e5a4558`。
- CK实验：`/workspace/ck-mellum-hf-korder`，`80f4da570`；独立JIT `/workspace/ck-mellum-hf-korder-jit`。
- 实验MoE `.so` SHA256：`cbf568d0928484e347d8571a30f2af3921ed2c9923237701a5921d266d31542e`。
- CK重放：`HIP_VISIBLE_DEVICES=1`，`PYTHONPATH=/workspace/ATOM-tritonmoe:/workspace/aiter-mellum-norm-hf`，对应`AITER_META_DIR/AITER_JIT_DIR/CK_DIR`，`CK_MELLUM_STAGE1_MODE=1`、`CK_MELLUM_STAGE2_ROUND_WEIGHT=1`、`ATOM_MELLUM_MOE_BF16_ROUTER_WEIGHTS=1`；固定A-0000、990-token prefill、L0 BF16张量，无在线请求或调度差异。
- 原始证据：`/workspace/logs/step0_continuation/norm990_active_prod/` 下 `ck_stage_isolation.json`、`fixed_router_replay_gpu1.json`、`expert55_lds_korder_probe.json`、`all51_korder_summary.json`、`local_split_remaining5.json`、`ck_stage_isolation_korder.json`、`ck_korder_first_difference.json` 及对应log/实际HF汇编。
- 初次CK lane选择实现产生大偏差，已拒绝并修正；失败日志保留为`ck_korder_rejected_lane_select.log`，未作为修复证据。
- 本轮未改变生产默认、未替换正在运行的服务、未推送。新方案服务性能未测，Stage2残余173值尚未修，A/C验收仍待完整修复后复测。


## 2026-11-03：生产 CK Up783 已接入，A10/C10×64 复测

最新生产状态以CK-PRODUCTION-UP783-FIX.md为准：Up783已在实际CK和8003服务修正；stage1最早首差现在为token1/e17/Up275。旧80f4da570实验不是活跃版本。当前CK仍Gate1387、Up1394不同，A10=20.46875%、C10=50%，Step0未通过。
