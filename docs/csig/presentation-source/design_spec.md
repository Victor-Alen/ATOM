# Mellum / ATOM 中文技术答辩 - Design Spec

## I. Project Information

| Item | Value |
|---|---|
| Project Name | Mellum 2.1 在 ATOM 上的正确性与推理优化 |
| Canvas Format | PPT 16:9，1280×720 |
| Page Count | 16 页：12 页主讲，4 页问答备页 |
| Design Style | pyramid + dark-tech，腾讯品牌蓝 |
| Target Audience | 腾讯 CSIG 计算加速技术面试官 |
| Use Case | 10 分钟主讲 + 10 分钟问答；全部中文，标识符保留原名 |
| Delivery Purpose | presentation |
| Content Strategy | 根据实测证据重组叙述；不得把小测、估算或历史版本当成最终全量数据 |
| Created Date | 2026-11-05 |

用户已在聊天确认推荐设计并授权自行迭代。主报告基于用户指定的 `OPTIMIZATION-REPORT.review-4pages.md` 更新。最终核心 `0453ef43` 的三轮完整测试、四行同版消融和三个profiling窗口已完成，源证据已导入 `sources/`；展示值与真实记录逐项核对。

## II. Canvas Specification

- viewBox：`0 0 1280 720`。
- 安全边距：左右 60，上 48，下 40。
- 主要内容区域：x=60..1220，y=145..630。
- 标题、证据出处、页码位置全篇一致；长路径放讲稿与附录，不挤入主讲图表。

## III. Visual Theme

- Mode：pyramid。先说明最终结果与边界，再展开验证、瓶颈、三个优化点、消融和交付。
- Visual style：dark-tech。深色负空间、细线、少量腾讯蓝高亮、等宽数字；避免全篇同一种卡片网格。
- 腾讯官方色彩规范第 3 页确认主色为 `#0052D9`，来源见 `../design-references/`。
- 参考腾讯官方 2026 年 9 月公司介绍的蓝色曲线、字体层级与留白；没有拿到可编辑内部母版，交付称“腾讯品牌风格”，不称“腾讯内部官方模板”。

| Role | HEX | Purpose |
|---|---|---|
| Background | #071323 | 全页底色 |
| Secondary bg | #10243C | 低对比内容面 |
| Primary | #0052D9 | 腾讯品牌主色、强调面 |
| Accent | #64B5FF | 深底上的高对比图线与关键数字 |
| Secondary accent | #7ED8D0 | 第二组数据及正确性通过 |
| Body text | #F6F9FF | 正文与标题 |
| Secondary text | #B4C5DA | 说明与次要标签 |
| Tertiary text | #8EA5C1 | 来源、页码 |
| Border/divider | #294361 | 分隔线 |
| Grid | #18314F | 图表辅助网格 |
| Warning | #FFB56B | 未达标或风险，避免仅靠颜色传递 |
| Scrim | #071323 | 如需覆盖层，与底色相同 |

仅封面允许用同色系透明度构成曲线纵深，不用炫光遮盖文字。腾讯蓝不用于深底小字；小字使用高对比浅色。

## IV. Typography System

| Role | Chinese | English | Fallback tail |
|---|---|---|---|
| Title | Microsoft YaHei | Arial | sans-serif |
| Body | Microsoft YaHei | Arial | sans-serif |
| Emphasis | Microsoft YaHei | Consolas | sans-serif |
| Code | Microsoft YaHei | Consolas | monospace |

- Title: `Arial, Microsoft YaHei`
- Body: `Arial, Microsoft YaHei`
- Emphasis: `Consolas, Microsoft YaHei`
- Code: `Consolas, Microsoft YaHei, Courier New`
- 字体已在本机核实；无额外字体安装要求。

| Slot | Size | Weight |
|---|---:|---|
| body | 32 | 400 |
| title | 48 | 700 |
| subtitle | 40 | 600 |
| annotation | 24 | 400 |
| footnote | 18 | 400 |
| lead | 36 | 500 |
| subheading | 36 | 600 |
| cover_title | 88 | 700 |
| hero_number | 96 | 700 |

尺寸全部为 unitless px。公式仅使用简短原生可编辑文字，不引入复杂 LaTeX。正文保持关键词与短句，完整推理放讲稿。

## V. Layout Principles

- 标题区 y=48..125；正文区 y=160..620；出处及页码 y=677。
- 封面以大字“正确性先行”与三段计算流形成视觉冲突，不做标题加三张卡。
- 主讲交替采用裸排数字、时间线、左右流程、非对称表格；数据页按信息量使用 dense。
- 卡片仅在并列信息确实需要容器时使用，圆角 6，间隔 24，内边距 24。
- 图表以零基线为默认，不同单位分开，吞吐向上更好、时延向下更好明确标出。
- 若最终 C 仍落后，用明确标记说明；不得以双轴或归一化图掩盖差距。

## VI. Icon Usage Specification

本稿不使用装饰性图标或 emoji。模块、箭头、流程全部为可编辑几何结构；腾讯名称以“腾讯 CSIG 校招技术答辩”表明使用场景，不暗示官方发布。

## VII. Visualization Reference List

Catalog read: 76 templates。参考模板位于 `C:/Users/蒋天颢/.codex/skills/ppt-master/templates/charts/`。

| Page | Template | Path | Summary-quote | Usage |
|---|---|---|---|---|
| P02, P08, P15 | basic_table | templates/charts/basic_table.svg | Pick for plain tabular text/number grid, 3-8 columns. Skip if cells need visual bars (use consulting_table) or qualitative scores (use harvey_balls_table). | 场景、消融与正确性表 |
| P03, P05, P07 | process_flow | templates/charts/process_flow.svg | Pick for 3-8 sequential steps connected by simple arrows — approval workflows, customer onboarding, request handling, lifecycle stages. Skip if cyclical (use circular_stages) or stages produce named outputs (use pipeline_with_stages). | 精度验证、MoE 路由、图捕获 |
| P04 | stacked_bar_chart | templates/charts/stacked_bar_chart.svg | Pick when each category splits into 2-4 internal parts and total still matters. Skip if only comparing totals (use column_chart). | 可核查 profiler 的算子耗时组成 |
| P09 | bullet_chart | templates/charts/bullet_chart.svg | Pick for 3-7 KPIs each with explicit target + actual. Skip for single metric (use gauge_chart) or items with only completion % and no target baseline (use progress_bar_chart). | A 吞吐与 C 两项时延对基线；B 无官方基线则单列数值 |
| P10 | pros_cons_chart | templates/charts/pros_cons_chart.svg | Pick for bilateral pros/cons list, 2-5 items per side. Skip for full feature comparison (use comparison_table) or numeric A/B mirror data (use butterfly_chart). | 选用与放弃的权衡 |
| P11, P16 | pipeline_with_stages | templates/charts/pipeline_with_stages.svg | Pick for 3-5 horizontal pipeline stages, each = title + 1-line description + output artifact, connected by arrows (data pipelines, ETL, build pipelines). Skip if any stage lacks an artifact (use process_flow or numbered_steps). | 仓库、启动、验证、证据产物 |
| P14 | timeline | templates/charts/timeline.svg | Pick for 3-8 milestone events on a horizontal time axis (no duration). Skip for tasks with start/end ranges (use gantt_chart) or vertical layout (use roadmap_vertical). | D1–D7 的真实决策与提交 |

Runners-up considered：P04 不用 pareto_chart，尚无可支持累计占比因果的排序论点；P09 不用 kpi_cards，已有明确基线；P14 不用 gantt_chart，日志记录里程碑而非精确持续时长。P06 SWA 的因果掩码结构不匹配现成模板，使用手工几何示意；不得伪装成实际 profiler 图。

## VIII. Image Resource List

无外部图片、无 AI 配图。官方 PDF 仅为设计参考，数据图表、算子图和时间线均保持可编辑。

## IX. Content Outline

### Slide 01 - 在正确性约束下提升推理效率

- Cover impact：以“正确性先行 / 三个优化点”作为核心冲突，左侧超大中文标题，右侧三段原生计算流穿过 BF16 舍入节点；腾讯蓝细线贯穿。
- Layout：非对称字形海报，非卡片封面。
- Info：蒋天颢；腾讯 CSIG 校招技术答辩；2026 年 11 月。
- Core message：优化收益必须与完整精度验收和可复现配置绑定。
- 时间：30 秒。

### Slide 02 - 三种场景使用不同的评价指标

- Core message：A 看每卡吞吐，B 看单流 TPOT，C 看长请求造成的尾时延。
- Content：单卡；A 512 请求、并发32；B 50 请求、并发1、输入1024；C 200 请求、泊松0.30次/秒；全部输出256。最终裁判隐藏集三次取中位数。
- Visualization：basic_table，3行场景+指标；基线只从 baseline.json 引用。
- 时间：40 秒。

### Slide 03 - 先修复数值边界，再恢复并行执行

- Core message：固定输入接近不等于自回归 token 一致，验收覆盖完整模型。
- Content：Norm 的 BF16 RNE、GEMM 的 shape/K-order、Attention 的舍入与归约顺序；逐算子→整模型→200条/16k/HumanEval。
- Evidence：当前冻结版本 C200 12800/12800，16k 一致，HumanEval 148/164 与 HF 相同。最终版本必须重新绑定证据。
- Visualization：process_flow，三层验证门槛。
- 时间：50 秒。

### Slide 04 - 瓶颈由真实时间线定位

- Core message：吞吐和尾时延的瓶颈不同，长 prefill 会阻塞 decode。
- Content：选取最终版本与对应对照的 torch profiler 原始 trace；展示主要 kernel 耗时与 launch 次数，注明采样负载和统计窗口。历史 trace 仅作为历史证据，不混入最终版本数字。
- Visualization：stacked_bar_chart；辅以原生时间段示意，注明示意而非截图。
- 时间：55 秒。

### Slide 05 - 优化点一：减少 MoE 的碎片计算与中间搬运

- Core message：围绕专家计算这一瓶颈，用 native CK/AITER 的路由与两阶段计算替代碎片路径。
- Content：64 专家、top-8、hidden2304、中间维896；复用路由、stage1/stage2 shape 选择、slot 输出、有序累加。仅列最终实际启用且通过验证的内容。
- Visualization：process_flow，输入→路由→专家→有序合并。
- Evidence：展示同一优化点的独立开关、预估依据和同条件实测；不把所有无关小算子归入此点。
- 时间：60 秒。

### Slide 06 - 优化点二：只计算注意力真正需要的区域

- Core message：滑窗与数值等价约束共同决定可省掉的 QK、softmax、PV 工作。
- Content：21 层1024滑窗、7层全注意力；保留 BF16 舍入边界；按冻结库的形状选择参数；未知形状安全回退。最终长上下文方案以完成验证的版本为准。
- Layout：大幅因果矩阵与1024带状窗口，右侧两条关键约束；不是装饰卡片。
- 时间：60 秒。

### Slide 07 - 优化点三：降低重复启动与图捕获成本

- Core message：将可捕获的稳定计算和动态元数据分离，减少重复 CPU 提交。
- Content：piecewise 编译、decode 图重用、稳定缓冲与索引；区分冷捕获、同进程预热与正式计时；验证图批次覆盖和 token 一致。
- Visualization：process_flow，准备元数据→捕获/复用→回放→输出。
- 时间：55 秒。

### Slide 08 - 累加消融说明每个点的独立贡献

- Core message：收益来自同条件逐项开启的测量，不把独立测试加速比相乘。
- Content：严格4行 Step0、+点1、+点1+2、+点1+2+3；列卡数、A吞吐、B TPOT P50、C TPOT P99/TTFT P99；注明请求数、重复次数、精度状态。
- Visualization：basic_table。采用最终同版四行消融，明确小测规模，不拼接历史版本。
- 时间：60 秒。

### Slide 09 - 全量结果逐项对照题目基线

- Core message：对 A、B、C 分别报告结果，明确尚未达标的指标。
- Content：最终候选三次全量中位数；基线 A638.2827、C TPOT41.2650ms/TTFT993.5490ms；B未提供基线。另注所有请求的成功数和输出长度。
- Visualization：bullet_chart，不同单位独立标尺；B旁置独立读数。
- 时间：55 秒。

### Slide 10 - 三个点的预算决定了方案取舍

- Core message：选择可验证的单卡收益，放弃缺乏完整证据或引入额外系统复杂度的方向。
- Content：MTP 作为研究对照，默认关闭；双卡按卡数归一化且须复验；量化改变精度门槛；其他缓存/调度候选若未纳入最终三点则明确排除。说明实测失败和放弃原因，不用理论估算代替运行结果。
- Visualization：pros_cons_chart。
- 时间：50 秒。

### Slide 11 - 从仓库到服务只需一条启动命令

- Core message：提交代码、开关、依赖、日志和评测命令形成可重复执行的交付。
- Content：最终 GitHub 分支与提交；`bash scripts/start_server.sh`；三个优化点开关；manifest、真实生成检查、评测产物。最终路径以实际提交版核实。
- Visualization：pipeline_with_stages，代码→服务→评测→证据。
- 时间：45 秒。

### Slide 12 - 代码与结论都有可追溯证据

- Closing impact：一句“能复现的收益，才计入最终结论”作为大字收束，下方两条证据链说明 AI 辅助与工作日志；并显示“进入10分钟问答”。
- Core message：AI 帮助开发与诊断，正确性、性能和提交状态由实际运行确认。
- Content：AI 工具用途、发现并纠正的错误建议；D1–D7 日志与真实提交对应。备页提供细节。
- 时间：40 秒。前12页合计600秒。

### Slide 13 - 备答：AI 使用与错误建议的验证

- Core message：错误建议通过受控实验识别，而不是通过文字可信度判断。
- Content：工具及用途、至少两条有证据的错误建议、发现方式、修正方法；不补造未使用工具。
- Layout：三行“建议→证据→决策”，可答约2分钟。

### Slide 14 - 备答：D1–D7 的真实决策过程

- Core message：保留问题、转向与失败记录，时间安排可以由提交和日志核查。
- Content：实际日历映射来自 Git/运行日志；如果跨出七天，单列补充日期，不倒填。
- Visualization：timeline，7个里程碑，附提交短哈希。

### Slide 15 - 备答：正确性门槛与反例

- Core message：局部张量误差、首64token一致率和HumanEval是不同的证据层级。
- Content：200条>=90%；16k单独统计；BF16 HumanEval gap<=2pp；MTP开启才要求on/off完全一致；仅单卡提交不声称双卡通过。
- Visualization：basic_table。

### Slide 16 - 备答：复现命令、环境与测量边界

- Core message：同代码、同依赖、同负载和同统计方式才能解释差异。
- Content：冻结版本哈希、启动参数、三个开关、A数据解析差异、热缓存与冷捕获界限、原始SQLite/trace目录。
- Visualization：pipeline_with_stages；长命令另附中文复现文档。

## X. Speaker Notes Requirements

- `notes/total.md` 及每页对应 `.md`：全中文可直接朗读的口语稿，不含计时或舞台提示；计时计划另存于 `10分钟讲稿.md`。
- 12页主讲合计600秒；另写10分钟问答准备，问答不强行连续播放备页。
- 所有数字绑定最终 evidence；未经完整验证的候选只能以“尝试/未采用”介绍。
- 对“为何不用MTP”“为什么C较差”“三个点是否合规”“精度为何对shape敏感”等问题给可辩护的答案。

## XI. Technical Constraints Reminder

- 逐页手写 SVG，主代理顺序执行；每页前重读 spec_lock。
- 文本以 tspan 排行，无 foreignObject、style、class、mask、script、动画。
- 颜色仅取 spec_lock；不使用 rgba；组透明度改为子元素透明度。
- 图表、标题与文本保持原生可编辑；PPTX完成后导出预览、检查溢出、字号、中文和页码。
- 所有数据页注明实测范围；不得将开源品牌参考说成腾讯内部母版。
