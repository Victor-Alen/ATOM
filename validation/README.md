# 公开集正确性复现

启动 ATOM 后执行 `/opt/venv/bin/python3 scripts/validate_public.py`。这会检查 C200 的前 64 token、严格 16384-token 输入及完整 HumanEval 164 题，并输出独立的 acceptance.json。该脚本仅是 HTTP 测试客户端，生产服务不读取这里的参考答案。

参考输出来自已归档的冻结 HF 运行，SHA256 与原始验收审计一致；不是本次命令重新运行 HF。HumanEval 使用模型原始 chat template、enable_thinking=False、greedy、EOS 停止、最大输出 2048；按完整程序提取后调用固定版本官方 checker。`vendor/` 保留 OpenAI HumanEval 的 execution.py 与 MIT 许可证，原始提交见 references/manifest.json。

`--screen` 只运行 C4 与严格16k，不运行 HumanEval，不是完整验收。`--verify-files-only` 仅做本地文件完整性与比较器负例检查，不占 GPU。

本次评估仅声明单卡、MTP关闭。出题方隐藏集与其他生成协议应由独立评测执行，不能以公开集参考复现替代。
