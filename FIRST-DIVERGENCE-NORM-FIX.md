# 首次 token 分叉与生产 RMSNorm 内核诊断（2026-11-03）

## L27 对齐的范围

此前通用 CK V3 修复对齐的是 prefill M=990 下 L27 的指定 up 标量（token467/slot3/expert38/feature132）：35.5→35.25，与 HF 相同。整张 up 张量仍有 1,154/7,096,320 个 BF16 值不同，并非整层完全一致。该修复作用于 V3；本次 decode block_m=32，实际走 V1。

## 原候选首个分叉：A-0000，第28个生成 token

固定共同的前27个输出，当前输入 token99，位置1016。HF 原生 eager BF16 cached teacher forcing 重新产生全部28个保存参考 token，证明该参考在此协议可复现。

- HF logits：1234与8411均32.25，HF argmax选1234。
- 生产 logits：8411=31.875，1234=31.625，选8411。
- HF LM head在生产同一隐藏状态上，98,304个输出与生产head逐值一致；生产head与sampler输入逐值一致。该步可排除head/sampler。
- 同一生产MoE输入、固定相同router IDs/weights，L27 MoE输出2304个值全部与HF一致。此结论仅限该步该层。
- 相同输入下router expert集合28层均一致；不能由此推出所有请求routing均无问题。
- 最终隐藏状态差max1.3828125、mean0.2202474。问题已在上游累积。

## 源码确认的额外差异：实际HIP RMSNorm

路径：ATOM layernorm → AITER rmsnorm2d_fwd(_with_add) → module_rmsnorm_quant → csrc/kernels/rmsnorm_quant_kernels.cu。并非另一个CKTile rmsnorm模块。

编译配置OPUS_FP32_to_BF16_DEFAULT=2，对gfx942使用截断。实际行为还包括残差F32相加后用未舍入F32值参与方差，以及归一化后未先BF16舍入就乘权重。

本步56个残差节点的输出全部逐值匹配RTZ；与HF的RNE残差输出合计36,309/129,024个值不同。基线真实kernel重放57个norm节点，完全复现捕获值，确认定位到了实际调用的kernel。

独立AITER分支修改真实kernel，只针对非量化BF16：残差和归一化后显式RNE舍入，最后乘权重结果显式RNE；不依赖改变全局Opus默认，不切换PyTorch参考实现。

| 固定输入算子重放 | 原生产kernel | 修正kernel |
|---|---:|---:|
| norm输出不同值/131,328 | 68,147 | 38 |
| 残差不同值/129,024 | 36,309 | 0 |

38个剩余值的具体归约根因尚未定位，不能声称norm全部bit-exact。

## 真实生产生成复测：仍未通过

与上一轮相同参考文件、同一生产attention和CK MoE，加入上述norm kernel候选。单客户端串行、冷启动、无reference/debug fallback。

| 场景 | 原候选 | norm修正候选 | 完全一致request |
|---|---:|---:|---:|
| A10×64 | 199/640=0.3109375 | 214/640=0.334375 | 1/10 |
| C10×8 | 66/80=0.825 | 57/80=0.7125 | 5/10 |

A-0000首次分叉从索引27变为索引3。局部误差减小不能推出自由生成一致率提高，不能据此宣称Step0通过。C此轮仅8个生成token，不能代表前64token通过。候选保留在独立分支和JIT目录，未替换原默认launcher。

## 版本与复现

- ATOM基底：e7199d66，分支dsh/mellum-first-token-divergence，诊断改动另行提交。
- CK：14e5a45589c8fbded83ac9fccce59fa317d240b3。
- AITER：9d4907d8d48640887264559bc01f8703a360c705，分支mellum/norm-hf-rounding，目录/workspace/aiter-mellum-norm-hf。
- JIT：/workspace/aiter-mellum-norm-hf-rne-jit。
- norm二进制SHA256：cf4a096d09fd1a6ae7730cd4ac095635333c8c070eea24e19792bf920a65bbca。
- MoE二进制SHA256：e7e648c5b818975b7d4e1eefd10b0b30e0c86bcce1db06d27c3df073c61b18e9。
- 候选启动：HIP_VISIBLE_DEVICES=0 bash scripts/start_mellum_norm_hf.sh。
- 服务level0/eager/BF16 KV/maxlen32768/util0.45/禁prefix cache；ordered MoE、router BF16 weights、stage1 mode1、stage2 round weight1。
- A参考SHA256：1f1bf047d422423076e102b800811723097340878adda3d275c3ebd6b6924fbd。
- C参考SHA256：daab8a306e4daf29dfc301555e3565b007b7d95c4f9e27b963172e2ce8644d47。
- 原分叉证据：/workspace/logs/step0_continuation/first_divergence/。
- 新生成结果：/workspace/logs/step0_continuation/norm_hf_ac10/。
- 未push。

## 修正norm后，新分叉继续拆到L0 attention

A-0000生成索引3（第4个token）的HF原生cached eager参考已重算：26828，top1间隔0.25；生产23213。共同输入token37、位置992。

- L0 input norm：生产/HF 2304个值完全一致。
- L0 attention输出：1473/2304不同，max0.0009765625、mean8.460124e-5。
- 在生产相同输入上单独重放投影：当前decode K/V投影完全一致；Q投影仅1/4096个值不同，max0.000244140625。
- 当前Q norm/K norm与HF同输入计算完全一致；prefill的全部4,055,040个Q norm及506,880个K norm值也在同输入下完全一致。
- 输出投影在同一attention输入上逐值一致（prefill与3个decode步）。
- prefill Q/K/V投影仍分别343/4,055,040、33/506,880、127/506,880个值不同，不能宣称投影完全bit-exact。

上述将后续问题进一步缩到attention内部及历史K/V轨迹；不应继续只盯L27 up。原来“attention不是误差源”的说法只能覆盖当时固定输入的特定算子测试，不能排除整条生产attention/KV链路。

固定实际生产旋转后的Q/K和历史V，在HF原生eager attention中重放，当前decode输出仍有1,178/4,096个BF16值不同，max0.00048828125、mean2.735175e-5。当前decode Q/K的RoPE结果本身与HF同输入逐值一致；prefill RoPE仍有Q273/4,055,040、K46/506,880个值不同。

这个实验把投影与当前RoPE误差隔离出去，剩余差异位于生产attention计算/实际KV读写路径。实验未直接逐值导出paged KV缓存，尚不能进一步断言只由softmax/归约造成。不能再把“attention不是误差源”当作全链路结论。

下一步明确为：读取并验真实际paged KV，然后在固定Q/K/V下拆QK、scale、softmax舍入、PV和split reduce；不要先继续修改L27 up。Step0仍未通过；本轮未测norm修复后的吞吐。

诊断重放：先启动候选并同时设置ATOM_LAYER_TRACE_DIR、ATOM_STEP_TRACE_DIR、ATOM_STEP_TRACE_STEP=3、ATOM_STEP_TRACE_L0_DETAIL=1；服务ready后才创建trace目录的ARMED，单请求生成4token。HF使用MELLUM_TRACE_ROOT和MELLUM_TRACE_STEP=3运行scripts/hf_first_divergence.py；随后运行analyse_first_divergence.py、probe_l0_attention_parts.py、probe_l0_attention_aggregation.py。诊断不改变模型数值计算。
