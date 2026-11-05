# 最终完整测试与原始证据

完整测试核心 `0453ef43`，单卡MI308X、BF16、MTP关闭，冻结依赖。后续材料提交不改变模型字节与实际运行开关。以下是公开集结果，不是隐藏集成绩。

| 指标 | 三轮中位数 | baseline.json | 结论 |
|---|---:|---:|---|
| A512 token/s/GPU | 699.1678 | 638.2827 | 提升9.54% |
| B50 TPOT p50 ms | 9.2401 | 未提供 | 不推算加速比 |
| C200 TPOT p99 ms | 64.5111 | 41.2650 | 仍未达到 |
| C200 TTFT p99 ms | 3948.2632 | 993.5490 | 仍未达到 |

每场景3轮，2286请求全部成功、全部流式输出256 token。先计算每轮百分位，再取三轮指标中位数；详细轮次见 `full-performance-summary.json`。C200为12800/12800、原严格16K为64/64、HumanEval148/164且164条序列与HF一致。另3条新16K输入串行/并发共384/384一致。数据与配置见 `correctness-acceptance.json`、`final-delivery-selection.json` 和归档。

服务选择HIP1，GPU显存实际分配位于PCI 0000:22:00.0；同机另一卡存在独立服务，不称为整机空闲测量。正式空闲环境和隐藏集仍待裁判复测。

## 解包最终与历史证据

```bash
mkdir -p results/evidence/final-current results/evidence/full-current results/evidence/supplemental results/evidence/late-control
tar -xzf results/final-acceptance-evidence-20261106.tar.gz -C results/evidence/final-current
tar -xzf results/portable-full-evidence-20261106.tar.gz -C results/evidence/full-current
tar -xzf results/delivery-supplemental-evidence.tar.gz -C results/evidence/supplemental
tar -xzf results/late-control-evidence-20261106.tar.gz -C results/evidence/late-control
```

最终归档SHA256：`976e806f587684d41eb0243c88a11e339b462613fdc2458cf37fac4665cf0fb3`；逐成员SHA256见同名manifest及 `docs/csig/evidence/final-acceptance-member-audit.json`。

- `final-current/model/`：本版完整精度、三轮A512/B50/C200、逐请求SQLite与审计、源码快照、启动manifest；screen目录单独标明小测。
- `final-current/ablation/`：本版四行累加消融，A64/B8/C12各一轮，每行先过832 token小门槛；不是全量消融。
- `final-current/profiling/`：本版3个torch profiler窗口、原始trace、cases与digest；累计GPU事件时间不是HTTP墙钟。
- `final-current/additional-long-context-hf/`：3条新增16K参考、HF生成参数/源码哈希、历史锚点与重复检查。
- `final-current/experiments/`：融合、warp、三角任务映射的算子记录与脚本；失败候选没有进入默认配置。
- `final-current/environment.json`：实际包版本、设备清单、打包依赖哈希及运行环境。
- `full-current/`、`supplemental/`：首版73d的完整结果、过滤A补充、旧消融/profiler与算子微基准，仅作历史。
- `late-control/`：lane-cache完整控制组和三种原生prefill失败消融。

汇总 `cumulative-ablation.json`、`current-profiling-digest.json` 与最终归档同版。未将旧的过滤A结果继承为本版成绩。失败原始融合核单独保存在 [实验分支](https://github.com/Victor-Alen/ATOM/tree/codex/csig-native-prefill-experiment-20261106)。

## 从头复测

```bash
bash scripts/start_server.sh
/opt/venv/bin/python3 scripts/validate_public.py
bash scripts/benchmark.sh --repeat 3 --out results/my-full-run
/opt/venv/bin/python3 scripts/summarize_benchmark.py results/my-full-run
```

输出目录使用新目录。`--screen --repeat 1`仅小测，审计时另加 `--expected-repeats 1`。汇总工具检查请求数、失败、流式输出长度和生成参数；精度与性能独立验证。
