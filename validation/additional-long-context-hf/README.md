# 补充的三条严格16K参考

从题目公开C数据的第0、49、199条原文构造输入：重复原文，按token前缀截取到16384，再decode/re-encode确认完全一致。这些输入只用于额外正确性检查，不替换性能数据。

HF参考采用冻结Mellum源码、BF16 eager、greedy、64输出、忽略EOS。生成前重新检查历史C0000和原严格16K两个锚点，生成后对第1条做一次同进程重复；该重复不是全局确定性证明。版本、源码和文件哈希见manifest.json，生成脚本保留在最终实验归档。

验证客户端对3条输入各运行串行一次、并发3一次，共6请求、384输出token。独立输入数量是3，不是6。参考答案仅供客户端比较，模型和服务端不读取此目录。

```bash
/opt/venv/bin/python3 scripts/validate_additional_long_context.py --references validation/additional-long-context-hf --out results/my-extra-16k
```

输出目录须为新目录。它不是完整C200/HumanEval门槛的替代，也不是性能测试。
