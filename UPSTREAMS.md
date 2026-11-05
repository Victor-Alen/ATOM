# 开源来源与本项目修改边界

本项目是在题目冻结环境上适配Mellum并修改相关算子，不将上游框架、算子库或继承的提交视为本项目独立成果。

| 来源 | 冻结基点或出处 | 本项目使用与修改 |
|---|---|---|
| [ROCm / ATOM](https://github.com/ROCm/ATOM) | 首个Mellum接入提交05a095b6的父提交：26f23a0bdbf50b9010c07ee3a71171189723e767 | 模型接入、请求/缓存适配、算子分派、编译与Graph、OpenAI服务及验证交付入口 |
| [ROCm / AITER](https://github.com/ROCm/aiter) | e3940660b40f4764cdf09147af96a2a764f264be | RMSNorm/BF16边界、分页decode attention及原生算子封装；全局/窗口QK的MFMA计算组织参考pa_decode_gluon.py |
| [ROCm / Composable Kernel](https://github.com/ROCm/composable_kernel) | af7118e342580ecd3f71edce7b1d0ba465012ecf | 两阶段MoE的累加次序、中间BF16转换、路由权重和独立slot输出；在原始CK kernel基础上改动 |
| 题目冻结的HF Mellum模型实现 | reference/hf_mellum/*.py中的版权声明；验证manifest中的文件SHA256 | 仅用于HF参考生成与数值语义核对；未升级Transformers环境 |
| 冻结PyTorch、hipBLASLt与Triton/Gluon | environment.json、runtime-manifest及形状查询工具记录 | 使用提供的算子/API及算法查询结果，保留库版本；未将底层原语声称为独立实现 |

AITER/CK的LICENSE随打包源码保留，路径分别为运行库的sources/aiter/LICENSE、sources/ck/LICENSE及相应分支源码目录。各修改分支头、未提交工作区patch、完整最终源码和逐文件SHA256见deployment/runtime-manifest.json；增量bundle、patch-series、commits.tsv与基点见deployment/history/。

上游与本项目修改可以通过完整Git父子关系和逐文件差异区分。patch-series中可能包含继承提交，不能把其中每行都算作本项目新增。交付默认路径以scripts/server_profiles/fast.json的开关与源码哈希为准，研究/失败分支不代表默认启用。
