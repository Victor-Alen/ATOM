# 当前完整测试与原始证据

核心实现 `73d21d03`，冻结原生依赖，单卡BF16、MTP关闭。以下是已完成的公开集结果，不能写成隐藏集成绩。

| 指标 | 三轮中位数 | baseline.json | 结论 |
|---|---:|---:|---|
| A512 输出 token/s/GPU | 697.7380 | 638.2827 | 提升9.31% |
| B50 TPOT p50，ms | 9.2052 | 未提供 | 不推算加速比 |
| C200 TPOT p99，ms | 68.3301 | 41.2650 | 仍未达到 |
| C200 TTFT p99，ms | 4387.3970 | 993.5490 | 仍未达到 |

每场景3轮，共2286条主测试请求全部成功、全部流式输出256 token；每轮先计算百分位，再对三轮指标取中位数。`full-performance-summary.json` 保存完整结果。过滤A输入的额外三轮是单独补充，不能替换原始A512主结果。

完整正确性见 `correctness-acceptance.json` 和 `portable-public-validation.json`：C200 12800/12800，严格16K 64/64，HumanEval148/164，与同协议历史HF参考相同，全部164条输出序列一致。参考协议、模型路径和数据哈希见相应记录。完整验证客户端实际复测也已经归档。

## 原始归档

归档文件的SHA256和来源见对应的 `.tar.manifest.json`。解包方法：

```bash
mkdir -p results/evidence/full-current results/evidence/supplemental
tar -xzf results/portable-full-evidence-20261106.tar.gz -C results/evidence/full-current
tar -xzf results/delivery-supplemental-evidence.tar.gz -C results/evidence/supplemental
```

- `full-current/performance/`：12组客户端结果，包括原始A、补充A、B、C；逐请求SQLite、summary、percentile、运行命令、请求审计和输入哈希。
- `full-current/correctness/`：C200实际输出、严格16K结果、HumanEval实际输出、HF参考审计、源码快照。
- `full-current/operators/`：QK/PV/softmax边界长度、负例与40次重复实验。
- `supplemental/delivery_profile_20261105T183352Z/`：使用完整打包依赖环境采集的3个torch profiler窗口及原始trace。
- `supplemental/cumulative_ablation_20261105T180217Z/`：四行累加消融，统一A64/B8/C12，每配置1轮，先过832 token小门槛。
- `supplemental/portable-public-validation/`：仓库验证客户端的完整公开集复测。

`cumulative-ablation.json` 是小规模消融汇总；`current-profiling-digest.json` 是按kernel名称分类的GPU事件累计时间，不是无profiler的HTTP时延。原始归档与解包副本的413个文件已逐一比对，SHA256见 `docs/csig/evidence/archive-member-audit.json`。

## 从头复测

```bash
bash scripts/start_server.sh
/opt/venv/bin/python3 scripts/validate_public.py
bash scripts/benchmark.sh --repeat 3 --out results/my-full-run
/opt/venv/bin/python3 scripts/summarize_benchmark.py results/my-full-run
```

输出目录应为新目录。`--screen --repeat 1`仅运行小测，审计时加`--expected-repeats 1`。新工具会检查请求数、失败、流式输出长度和生成参数，缺轮或不完整时不输出“完整通过”。精度与性能是两次独立验证。
