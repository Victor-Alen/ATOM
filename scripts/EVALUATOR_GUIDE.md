# 评测复现入口

请在题目冻结的 Linux 容器内执行，不升级 ATOM / AITER / ROCm / PyTorch / Transformers。

```bash
bash scripts/start_server.sh
bash scripts/benchmark.sh --repeat 3
```

性能基线取 `/data/datasets/baseline.json`。A 比较输出吞吐/GPU；C 比较 TPOT p99 和 TTFT p99；B 基线未提供。

正式性能每轮 A512、B50、C200，三轮取对应指标中位数。`--screen` 只用于 A64/B8/C12 筛选；报告会明确区分。累计消融的启动配置为 `--ablation step0/point1/point12/point123`。

EvalScope 的 A 数据集读取适配、原始 512 条与过滤补充结果的区别、正确性协议及证据位置见 [交付说明](../README-CSIG.md)。默认不使用 MTP。公开测试不代替出题方的隐藏测试。
