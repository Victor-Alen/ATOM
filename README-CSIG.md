# Mellum 2.1：ATOM 单卡适配与三点优化

这是腾讯 CSIG 云一计算加速题目的交付入口。模型为 Mellum2.1-12B-A2.5B-Thinking，运行于题目冻结的 ROCm / PyTorch / AITER 环境和单张 MI308X。默认 **三个优化点开启，MTP 关闭**。本项目保留完整提交历史，不 squash。

## 一条命令启动

在题目给定的 Linux 容器内进入本仓库，执行：

```bash
bash scripts/start_server.sh
```

脚本校验源码和打包依赖的 SHA256，首次启动自动解包 `deployment/`，加载模型、预热及捕获 decode graph，最后执行一次真实的 8-token 生成。只有生成成功才打印 `READY`。模型权重继续使用 `/data/models/Mellum2.1-12B-A2.5B-Thinking`，不下载权重、不升级环境。

默认 HTTP 地址为 `http://127.0.0.1:8000/v1`。需要其他端口或对外监听时：

```bash
bash scripts/start_server.sh --gpu 1 --host 0.0.0.0 --port 8000 --engine-port 18000
```

`--gpu` 是冻结容器内的 HIP 设备选择。本次实测为 `ROCR_VISIBLE_DEVICES=0,1`、`HIP_VISIBLE_DEVICES=1`，物理 PCI 总线 `0000:22:00.0`，计为 **1 张卡**；不能仅根据容器的逻辑序号推断物理卡。脚本不抢占、不结束其他服务。已有服务占用端口或显存时应先选择空闲资源。

## 三个优化点及开关

| 优化点 | 解决的瓶颈 | 单独开关 | 主要实现 |
|---|---|---|---|
| 点 1：MoE 专家计算 | HF 形状兼容路径中的专家权重搬运、碎片化 GEMM、路由和输出归并 | `--moe-opt 0/1`；环境变量 `OPT_MOE` | 原生 CK 两阶段专家核、按形状选择归约次序、BF16 路由融合、独立 slot 输出、路由复用 |
| 点 2：Attention 计算 | SWA 层对窗口外位置的无效计算与读写，以及 decode attention 的小算子启动 | `--attention-opt 0/1`；环境变量 `OPT_ATTENTION` | AITER/Gluon decode 核、窗口 QK、保持 HF 归约顺序的 softmax、窗口 PV |
| 点 3：分段编译与 Graph 执行 | 小算子调度、Python/CPU 提交和重复 kernel launch | `--compile-graph-opt 0/1`；环境变量 `OPT_COMPILE_GRAPH` | piecewise compile、稳定 decode 元数据与 ATOM decode graph；关闭时强制 eager |

关闭某一点只回到其正确性兼容实现；Norm、Linear、RoPE 的适配与 BF16 精度修正属于共同的 Step 0 基础，不重复记作新的优化点。形状、dtype、布局或冻结版本不满足已验证条件时，相关算子回退到兼容路径。

累计消融使用以下配置，每次切换先正常停止本仓库启动的服务：

```bash
/opt/venv/bin/python3 scripts/stop_server.py --port 8000
bash scripts/start_server.sh --ablation step0
# 另外三个配置：point1、point12、point123；默认 point123
```

独立关闭某一个点的示例：

```bash
bash scripts/start_server.sh --attention-opt 0
OPT_COMPILE_GRAPH=0 bash scripts/start_server.sh
bash scripts/start_server.sh --dry-run --ablation point12
```

不要把不同配置的性能混在一起。每次启动的 `manifest.json` 记录命令、展开后的环境变量、三个点的状态和 PID，默认位于 `/workspace/logs/mellum-start-server/`。

## 验证与测量

```bash
# 题目 A512 / B50 / C200；单卡；每场景 3 次
bash scripts/benchmark.sh --repeat 3

# 优化阶段的小规模筛选：A64 / B8 / C12
bash scripts/benchmark.sh --screen --repeat 1
```

A 使用并发 32、B 使用并发 1，C 使用开放环泊松到达率 0.30 请求/秒；均输出 256 token，`ignore_eos=true`、`temperature=0`、流式返回。C 的指标是 TPOT p99 / TTFT p99，不能拿 C 的吞吐代替。正确性检查与性能测量分开；HumanEval 的生成协议也不等同于 256-token 性能协议。

冻结环境的 EvalScope 1.14.1 不支持把文件名直接当作 `--dataset` 插件名。因此脚本把 A 的 `prompt` 字段原样写成请求体，使用 `line_by_line` 读取，不截断、补齐或改写提示。默认 `--a-source all` 保留全部 512 条原始记录；源文件有 5 条实际 token 数略超 4096。`--a-source filtered` 是单独的补充测量，不能与原始 512 条结果混称。`--stream` 是该客户端布尔参数的正确拼写，语义对应题目的流式开启。

以 `/data/datasets/baseline.json` 为唯一性能基线：A 为 **638.282724048549 token/s/GPU**；C 的 TTFT p99 为 **993.5489579802379 ms**、TPOT p99 为 **41.26502377250433 ms**。该文件没有提供 B 基线。不要把候选人的 Step 0 数据当成出题方基线，也不要把小样本结果写成全量结果。

最终数据、完整验收范围、消融样本量、未达标项见交付报告和 `results/` 内的证据索引。公开集通过不代表隐藏集已经验收。

## 源码和依赖位置

- `atom/model_ops/mellum_*.py`：精度兼容、算子封装、Gluon/Triton 核与形状分派。
- `atom/model_engine/model_runner.py`、`mellum_request_gemm.py`：Graph 生命周期与稳定元数据。
- `scripts/server_profiles/fast.json`：默认开关、固定命令、源码 SHA256。
- `scripts/query_hf_qk_profiles.cpp`、`query_hf_pv_profiles.cpp`：冻结 hipBLASLt 的全长度形状查询，不读取提示内容或答案。
- `deployment/runtime-manifest.json`：每个打包源文件、构建输入和本地原生二进制的来源及 SHA256。
- `deployment/runtime-gfx942.tar.gz.part*`：分片存放的运行依赖。解包后源码位于 `.runtime/<摘要>/sources/`，生成的 HIP/CK 实例、构建描述及二进制位于其 `runtime/`。

打包保留了修改后的 AITER/CK 源码、相关生成实例和构建记录；默认使用经验证的冻结二进制。可选的从源码重建属于独立复现过程，不能将“源码已打包”表述成“已在全新环境重建验证”。运行仍要求题目给定的 Python/ROCm 和基础 AITER 安装，跨 PyTorch、ROCm 或 GPU 架构不承诺二进制兼容。
