# 中文面试材料

此目录对应单卡 BF16、MTP 关闭的 Mellum / ATOM 三点优化。当前完整测试核心实现为 `73d21d03`；验证客户端和交付工具有后续提交，模型源码以启动 profile 的 SHA256 绑定。报告如实保留 C 时延未达到题目基线的结果。

| 材料 | 入口 |
|---|---|
| ≤4页技术报告 | [PDF](优化报告.pdf)、[可编辑正文](优化报告.md) |
| 20分钟答辩 | [中文PPT](Mellum_ATOM_中文答辩.pptx)、[16页预览](答辩PPT预览.pdf) |
| 10分钟主讲 | [逐页讲稿](10分钟讲稿.md)；12页主讲，另4页备查 |
| 10分钟问答 | [问答准备](答辩问答准备.md) |
| 1页AI使用说明 | [PDF](AI使用说明.pdf)、[正文](../../AI-USE.md) |
| 按天工作日志 | [WORKLOG.md](../../WORKLOG.md)、[提交追溯](evidence/worklog_git_history.current.json) |
| 代码与启动 | [仓库入口](../../README-CSIG.md)、[代码修改索引](代码修改索引.md) |
| 原始证据 | [结果归档与解包说明](../../results/README.md)、[总体证据索引](证据索引.md) |
| 演示源稿 | [SVG与讲者备注](presentation-source/) |

PPT为腾讯品牌蓝配深色技术版式的原创设计，不声称使用内部官方模板。图形和文字是可编辑PowerPoint元素，16页均含讲者备注；计时脚本为600秒的排练分配，没有冒充实际演讲录音或本人排练成绩。

本目录的报告与总体证据索引沿用完整交付包的相对命名：其中 `evidence/full-current/`、`evidence/supplemental/` 在Git仓库中对应解包后的 `results/evidence/full-current/`、`results/evidence/supplemental/`。无需解包也可直接查看 `results/` 中的汇总JSON。
