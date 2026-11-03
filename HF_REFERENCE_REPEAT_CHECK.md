# HF Mellum 参考输出重复性核查

2026-11-03。使用 `/workspace/ATOM-tritonmoe/scripts/hf_reference_generate.py` 的同一模型加载与生成方式：Transformers 5.2.0、bf16、`attn_implementation="eager"`、`do_sample=False`、`num_beams=1`、`use_cache=True`，同一进程内逐条重复。复核脚本 `/workspace/ATOM-tritonmoe/scripts/probe_hf_repeat_cached.py`，原始结果 `/workspace/logs/step0_continuation/hf_repeat_cached.json`。

| 样本 | prompt token | 生成长度 | 连续三次 | 与已存 eager 参考前缀 |
|---|---:|---:|---|---|
| A-0000 | 990 | 16 | 逐 token 相同 | 相同 |
| C-0001 | 12,580 | 8 | 逐 token 相同 | 相同 |

C-0001 三次均为 `[233,233,37,6539,233,889,2335,233]`。另外，`hf_scenario_c_first5_eager.jsonl` 与 `hf_scenario_c_200_eager.jsonl` 在重叠的 C-0000 至 C-0004 上，**5/5 条各 64 token 全相同**，包括 C-0001。

新探针分别把 `max_new_tokens=min_new_tokens` 设为 16 和 8；已存文件设为 64。因此新旧对照核验的是前缀，而三次重复之间使用完全相同的生成长度。

因此，在已测固定 eager、缓存生成配置和样本上，HF 参考输出是可重复的。此前 `RESULTS.md` round 66 和 `TASK-STEP0-CONTINUATION.md` 中“C-0001 HF 自己复现不出来”的无条件说法与这些新证据不符；当时的 fresh-versus-recorded 差异不能再当作固定 HF 配置非确定性的证据。旧实验的确切配置差异尚未在本次逐项复盘。

**配置不能混用。**已有 `sdpa` 与 `eager` 的两份 C-200 参考，仅 `3469/12800=27.10%` 的 token 位置相同；这是不同 attention backend 的数值路径差异，并非同一 HF 配置重复运行的随机性。`use_cache=True` 的自回归生成与 `use_cache=False` 的整段重算也必须分别标注。本次没有把全部 200 条在固定配置下重复生成两遍，因此不声称全量 200 条已通过重复性验收。
