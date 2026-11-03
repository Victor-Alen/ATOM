# 最新 CK up 修正版：A10 / C10 输出一致率

2026-11-03。固定顺序运行 A-0000～A-0009、C-0000～C-0009，各10请求，单客户端串行。A每条64 token，C每条8 token。两组均未达到0.90门槛。

| 场景 | 一致 token 位置 / 总位置 | 一致率 | 整串完全一致请求 | ≥0.90 |
|---|---:|---:|---:|---|
| A10×64 | 199/640 | **0.3109375** | 1/10 | 否 |
| C10×8 | 66/80 | **0.8250000** | 7/10 | 否 |

这里“一致率”按与HF生成token在相同输出位置是否相等统计，未将局部张量精度或请求成功率代入。请求全部正常完成。

| 请求 | 一致 / 总 token | 首次分叉位置（从0计） |
|---|---:|---:|
| A-0000 | 27/64 | 27 |
| A-0001 | 8/64 | 4 |
| A-0002 | 3/64 | 3 |
| A-0003 | 12/64 | 9 |
| A-0004 | 32/64 | 31 |
| A-0005 | 64/64 | 无 |
| A-0006 | 25/64 | 25 |
| A-0007 | 17/64 | 17 |
| A-0008 | 3/64 | 2 |
| A-0009 | 8/64 | 7 |
| C-0000 | 8/8 | 无 |
| C-0001 | 5/8 | 5 |
| C-0002 | 8/8 | 无 |
| C-0003 | 8/8 | 无 |
| C-0004 | 8/8 | 无 |
| C-0005 | 8/8 | 无 |
| C-0006 | 0/8 | 0 |
| C-0007 | 8/8 | 无 |
| C-0008 | 5/8 | 5 |
| C-0009 | 8/8 | 无 |

测试版本与配置：ATOM运行提交 `4e6855ae`，CK `14e5a4558`；二进制SHA256 `e7e648c5b818975b7d4e1eefd10b0b30e0c86bcce1db06d27c3df073c61b18e9`，与之前双卡数值验收版本相同。V3 up独立FP32块＋FP64合并、stage1 HF中间舍入、stage2 down舍入、逐slot定序聚合及路由权重BF16对齐均启用。没有启用HF/PyTorch MoE替换分支。

服务：端口8003，level0、enforce-eager、BF16 KV、max-model-len32768、gpu-memory-utilization0.45、prefix cache关闭；temperature0、ignore_eos=True。启动前atom.entrypoints进程数为0，GPU显存均为0%；运行时实际加载模型、GPU显存约42%～44%。

参考：A使用已核验保存的HF eager文件；C使用此前现场重算的HF eager/bf16/greedy/use_cache=True参考。启动前均与冻结manifest核对SHA256通过，此轮未重新生成参考。

- A：`/workspace/correctness/hf_scenario_a_20_eager.jsonl`，SHA256 `1f1bf047d422423076e102b800811723097340878adda3d275c3ebd6b6924fbd`。
- C：`/workspace/logs/step0_continuation/ck_round_fresh_hf_c20.jsonl`，SHA256 `daab8a306e4daf29dfc301555e3565b007b7d95c4f9e27b963172e2ce8644d47`。

复现服务启动：`ATOM_MELLUM_MOE_BF16_ROUTER_WEIGHTS=1 HIP_VISIBLE_DEVICES=0 MELLUM_SERVER_PORT=8003 scripts/start_mellum_up_fp64.sh`。客户端使用 `scripts/compare_atom_hf.py`，分别指定上述参考文件、`--limit 10`、`--max-tokens 64`或`8`、`--url http://127.0.0.1:8003/v1/completions`。

完整原始结果：服务器 `/workspace/logs/step0_continuation/up_fp64_ac10/`，包括server.log、a10/c10原始响应及逐请求比较。`run.json`保存测试提交、实际服务进程环境、参考/二进制hash及逐条HF/ATOM token ids。本地 `UP_FP64_AC10_RAW.json`、`UP_FP64_AC10_PER_REQUEST.csv`。

本轮仅测试上述A10/C10，不代表完整20/200条验收或三端口同批；B本轮未跑。现有证据仍不能宣称Step0通过。结果已写入RESULTS/HANDOVER。请求全部结束后服务正常退出，GPU handle owner=0，两张GPU显存均为0%。未推送。
