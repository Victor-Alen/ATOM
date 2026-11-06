# Mellum / ATOM 三点优化交付

正式三个点：Attention 有效窗口计算；piecewise compile 与 Graph；MTP K=1。
原生 MoE 的正确性移植属于公共 Step 0，额外 MoE 多流、路由融合等加速开关在四行消融中均关闭。

## 当前验收状态

核心源码固定于 `3ea8fc120ada383dc83a19a6ac57f0e8c755b32b`。新启动入口只是移植这套配置，没有更改模型计算代码。
同一 MI308X / PCI 0000:30:00.0 的四行 A32/B4/C6 小测和每行 832-token 检查已通过。
A512/B50/C200 各三轮已完成：2286 个性能请求全部成功。三轮中位数为 A 680.6955 tokens/s/GPU、B TPOT p50 6.2936 ms、C TPOT p99 61.0210 ms、C TTFT p99 4607.2047 ms。A 超过 baseline；C 两项仍未超过 baseline，不能写为全部达标。
MTP on/off 两臂各 C200 12800/12800、严格 16K、原版 HumanEval 148/164（与 HF 相同），另有 762 对 256-token 原始 ID 全部一致。独立完整审计通过，完整数据见交付证据目录。HumanEval+ 另行独立复评为两臂和 HF 均 142/164，不能与原版分数混用。
02a75c2b 资产入口已在实际 GPU 上以 MTP_DOWNLOAD=0 完成离线启动、8-token生成和320-token HF/严格16K检查；默认三个点开启、MTP K=1，实际加载原生库与完整测量核心一致。联网下载仍未验证成功，离线交付使用随包头文件。启动记录、三轮汇总及独立审计见 `docs/csig-validation-20261106/`。后续本次文档及状态说明修订没有更改计算或运行开关。

## 一条命令启动

在指定的冻结 Linux / ROCm 容器中，从仓库根目录执行：

```bash
bash scripts/start_server.sh
```

默认单卡 GPU0，HTTP `127.0.0.1:8000`，引擎端口 18000，三个点开启，MTP K=1。
只有模型预热完成并实际生成 8 个 token 后才打印 `READY`；日志与实际命令写入 `/workspace/logs/mellum-start-server/`。
脚本不会停止其他服务。已占用的端口会报错；停止自己此前启动的服务使用：

```bash
/opt/venv/bin/python3 scripts/stop_server.py
```

冻结环境必须已有 `/opt/venv/bin/python3`、MI308X/gfx942 依赖和提供的基础模型。
基础模型默认 `/data/models/Mellum2.1-12B-A2.5B-Thinking`，可通过 `MODEL_BASE_PATH` 指定。
MTP 头优先使用显式 `MTP_HEAD_PATH`，其次为仓库 `deployment/model-assets/model-mtp.safetensors`，再使用 `.runtime/model-assets/` 缓存。首次缺失时，启动命令自动从 Hugging Face 固定 revision 获取；连接失败可尝试 hf-mirror.com 的同一 revision，两者均强制校验相同长度和 SHA256。可用 `MTP_DOWNLOAD_BASE_URL` 指定 HTTPS 传输源。
离线验收时，将随交付提供的头文件放入 `deployment/model-assets/`；`MTP_DOWNLOAD=0 bash scripts/start_server.sh` 可禁止联网。脚本不依赖开发者的 `/workspace/mtp-coordination/` 目录。
头来自官方 `JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP` 的固定 revision `d55846102dd1c047e95d4841070f88290227c3b7`，SHA256 为 `89a2f2454b185b97b809eef2b2a2c32300c6cf550fdbde6430a1c2e8290e025c`。
Git 包含对应配置、索引和固定版本获取脚本，不包含 24GB 基础权重或 857MB MTP 头。基础权重由题目环境提供；MTP 头随本地交付另附，或按上述方式自动获取。不静默降级成 MTP 关闭。
入口在 `.runtime/model/` 建立模型视图，基础权重及 tokenizer 使用符号链接，不修改提供的权重。
冻结原生运行库和修改来源位于 `deployment/`，解包与 SHA256 检查由入口执行，不升级 ATOM、AITER、ROCm、PyTorch 或 Transformers。

可选择另一张已授权的空闲卡或端口：

```bash
bash scripts/start_server.sh --gpu 0 --port 8000 --engine-port 18000
bash scripts/start_server.sh --dry-run
```

## 三个独立开关与累计消融

| 配置 | Attention | compile/Graph | MTP |
|---|---:|---:|---:|
| `--ablation step0` | 0 | 0 | 0 |
| `--ablation point1` | 1 | 0 | 0 |
| `--ablation point12` | 1 | 1 | 0 |
| `--ablation point123`（默认） | 1 | 1 | 1 |

独立覆盖开关为 `--attention-opt 0/1`、`--compile-graph-opt 0/1`、`--mtp-opt 0/1`，对应环境变量 `OPT_ATTENTION`、`OPT_COMPILE_GRAPH`、`OPT_MTP`。命令行优先。
每次切换需先停止本仓库启动的服务，再重新启动。所有实际开关、环境、命令都记录在启动 manifest 中。
关闭 Graph 时自动使用 eager；关闭 MTP 时移除投机解码参数并关闭 MTP 专用算子与 graph metadata。MTP 只支持本次验证的 K=1、TP=1、PP=1。
四行共享相同 MoE 正确性路径、预填充调度和精度边界，不能把恢复原生 MoE 单列为第四个优化点。
独立开关的其他组合不是四行测量的替代，不能自动继承验收结果。

## 正确性与完整性能命令

服务启动后，以下工具自动从 `/v1/models` 读取实际模型名称：

```bash
/opt/venv/bin/python3 scripts/validate_public.py --mtp on
bash scripts/benchmark.sh --repeat 3 --mtp on
```

正确性脚本检查 C200 前 64 个 token、严格 16,384-token 输入和 HumanEval164；`--mtp` 仅是实际运行状态的记录，不能代替 MTP on/off 的逐 token 对比。
完整 MTP 开关对比必须保留两臂原始输出，对同一输入的 token ID 比较；本次全量原始数据已在本地交付的 `evidence/full3-complete-archives/` 归档，独立审计摘要同时随本仓库交付。
性能为 A512 并发32、B50 并发1且输入1024、C200 泊松到达0.30请求/秒；均输出256、greedy、ignore_eos、stream。
三次先分别计算各轮吞吐/p50/p99，再取三轮中位数，不对逐请求数据混池后冒称中位结果。
`--screen` 只用于开发小测，不能写成完整测试通过。

给定 EvalScope 版本要求注册的数据集名称；A 的 JSONL 从每条记录提取原始 prompt 字段，通过 `line_by_line --dataset-path` 传入，保持全部512行内容和顺序，记录原始哈希。不额外按 tokenizer 长度删除数据。
官方 `baseline.json` 的 A 为638.282724048549 tokens/s/GPU，C TPOT p99为41.26502377250433ms、TTFT p99为993.5489579802379ms；没有独立的场景 B 基线。
本项目 Step 0 只是累计消融的起点，不能替代官方基线计算得分。最终隐藏集结果仍由裁判测量。
