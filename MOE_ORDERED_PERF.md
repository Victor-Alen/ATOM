# Mellum MoE 逐 slot 定序聚合：两个输入的性能小测

2026-11-03，GPU0（MI308X/gfx942），ATOM `791170bb`，AITER `8ddfc7510`。真实 L0 输入、64 expert bf16 权重和固定 top-8 routing。比较当前生产 `aiter.fused_moe` 与同 CK stage2 GEMM 的逐 slot 输出、GPU 索引重映射和 fp32 顺序累加。

每种模式预热 10 次，再交替执行并同步计时各 50 次；下表为完整 **MoE 调用** 的墙钟延迟中位数，包含 sorting、stage1、stage2、分配及替代路径的重映射/reduce，不包含 gate GEMM、top-k routing 或模型其他层。

| 输入 | 当前融合 MoE | 逐 slot 定序 | 延迟倍数 | 等量调用吞吐变化 |
|---|---:|---:|---:|---:|
| decode，M=1 | 0.230 ms | 0.433 ms | **1.89×** | 约 **−47%** |
| prefill，M=866 | 1.175 ms | 1.430 ms | **1.22×** | 约 **−18%** |

M=1 的 p10–p90：原路径 `0.200–0.241 ms`，定序 `0.412–0.446 ms`。M=866：原路径 `1.144–1.190 ms`，定序 `1.410–1.446 ms`。这是探针实现的核调用开销；它目前没有接入生产服务，**不能据此声称服务 tok/s 降低相同比例**。本实现采用多次普通 PyTorch GPU 操作完成索引变换和 8 次 fp32 累加，尚未做融合优化。

脚本：`/workspace/ATOM-tritonmoe/scripts/bench_moe_slot_stage2.py`。原始 50 次样本和统计：`/workspace/logs/step0_continuation/bench_slot_stage2_gpu0.json`。替代路径的输出哈希与之前固定 stage1 实验一致（M=1 `583fec11fe6764c9`；M=866 `e0699e24c2f5dcb2`）。
