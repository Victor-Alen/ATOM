# Step 0 五项口径复核 — 2026-11-04

生成时间：2026-11-04T12:23:39.868233+08:00。

**按用户最终选择的单卡 BF16、MTP 关闭配置：通过本轮基础 Step 0 验收。**

这不是“MTP 开启或 TP=2 也已通过”的结论。题目允许单卡完成 Step0；MTP 与双卡条款是启用对应配置后的附加要求。开启 MTP 与 TP=2 的实际失败记录均保留。

冻结计算代码 `a787ed595fd4a0c50b9c54aff2a98de3961c26b8`；测试开始时仓库提交 `680591d58f968a944f9f84c1c319265641b0cdc8`。没有改动计算源码、放宽逐 token 比较或缓存模型输出。

| 项目 | 实测结果 | 当前配置判定 |
|---|---|---|
| 1. 200 条公开 prompt，前64 token ≥90% | 12800/12800 = 100.00%；整段一致200/200 | 通过 |
| 2. HumanEval BF16 pass@1 差距≤2个百分点 | HF 148/164 = 90.24%；ATOM 148/164 = 90.24%；绝对差 0.0000 个百分点 | 通过本轮配对协议 |
| 3. 开启MTP与关闭逐token完全一致 | 开启探针启动失败，KeyError: MellumForCausalLM；没有完成开关输出比较 | 用户选择关闭，本次不适用；开启能力不能标为通过 |
| 4. 16k 输入单独统计 | 原公开集约16k组11条，704/704；另加严格16384-token输入，64/64 | 通过 |
| 5. 若采用双卡并行，也须通过 | TP=2启动探针触发TP=DP=EP=1断言；其他正式双卡并行配置未验收 | 交付配置单卡，本次不适用；不宣称双卡能力通过 |

## HumanEval 配对参考的具体范围

用户确认重测HF与ATOM，建立相同模板与配置的配对参考。两边均使用：官方聊天模板、`enable_thinking=False`、greedy、EOS=28停止、max_new_tokens=2048；原始164道HumanEval题；模型生成完整Python程序后统一提取代码，原始测试和entry_point不变。

采用OpenAI官方测试器（版本 `b9453e8f1865df0a0057a1383767caa400acfbcf`），每题3秒测试超时，独立CPU子进程4GiB内存上限。全部164道官方标准答案先通过测试器自检；空程序、错误实现和死循环三个负对照均按预期失败或超时。没有把公开测试或标准答案放入模型提示。

测试器来源：[OpenAI human-eval execution.py](https://github.com/openai/human-eval/blob/b9453e8f1865df0a0057a1383767caa400acfbcf/human_eval/execution.py)。

HF截断题数 3，ATOM截断题数 3；所有题均保留在164题分母中。两边完整生成token序列一致 164/164，提取后的程序一致 164/164。

**这是本轮明确协议下的配对HF参考，不是模型卡HumanEval+ 91.5%的复现，也不混用早期raw prompt/256-token得到的4/164结果。** 本轮没有启用FP8，未验收FP8的5个百分点阈值。

## 公开集与长上下文

公开200条来自`/data/datasets/scenario_c_prompts.jsonl`，与冻结HF eager参考逐条prompt文本一致。ATOM本轮在同一单卡服务重新生成全部200×64；HF另外现场重算C0000、C0049、C0199，三例均逐token复现冻结参考。

原始约16k分组的预先定义区间是15360–16384个实际token，最大16019；共11条，逐条结果见审计JSON。严格16384-token探针是在公开C0049后添加填充构造的**额外测试**，不冒充原始公开集样本，tokenizer重编码与API usage均确认为16384。

HF参考和ATOM分别使用分配的两张卡；正式ATOM模型始终是同一个TP=1实例（port8005）。原先评测副本分片计划已在启动前取消；164道ATOM题都由同一实例完成。

## 失败能力探针和历史更正

- 当前模型权重索引只含28层主模型，没有MTP/draft/nextn类张量；本机模型README说明MTP head仍待发布。开启`--method mtp`实际报架构注册KeyError，不能用“关闭MTP的输出稳定”替代开关一致性验证。
- TP=2实际报MoE后端TP=DP=EP=1限制。这个结果只说明本候选TP=2不能启动，不推广成所有DP/EP/P-D方案都不可行。
- 旧报告把16019-token OOM样本写作C0059，是把服务请求编号当成数据集编号。按公开集实际分词长度，唯一对应样本是C0049；这次更正ID。既有token一致率与原始日志不变。
- 之前“Step0已通过”的概括应理解为当时token一致性阶段通过；本报告才给出HumanEval、严格16k和条件项的分项结论。

## 证据与复现

根目录：`/workspace/logs/step0_five_gate_20261104`。独立汇总：`five_gate_independent_audit.json`。配置：`selected_configuration.json`和`runtime_manifest.json`。原始结果：`public200_single_gpu/`、`he_hf.jsonl`、`he_atom.jsonl`、`strict_16k_*.json`；失败探针：`configuration_probes.json`及两个startup.log。

所有请求走ATOM OpenAI兼容server；HF仅作为离线参考，没有给ATOM注入参考hidden states或logits。未做吞吐性能、A512/C256输出长度或正式双卡并行性能验收。
