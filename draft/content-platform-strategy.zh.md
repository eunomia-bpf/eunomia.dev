# 长期内容平台与个人技术品牌运营草案

> 状态：内部草案，不发布。最后更新：2026-07-19。
> 本文是战略运营文档，不是 agent skill。skill 只保存可重复执行的流程、边界、平台 QA 和校验。

## 核心判断

长期内容平台应该服务个人技术品牌，而不是只服务 eunomia.dev 的搜索排名。eunomia.dev 是可信档案馆和作品集；GitHub、论文、demo、talk 是证据资产；X、LinkedIn、知乎、掘金、Reddit、HN、Lobsters、Medium、Dev.to 是平台原生讨论和传播场。

SEO/GEO 是帮助内容被发现、引用和复述的技术手段，不是最高目标。最高目标是让目标读者在想到 eBPF、AI-agent infrastructure、runtime observability、systems safety、GPU/runtime research 时，能想起维护者的判断力、研究 taste、开源可信度和工程实践能力。

## 内容发布 skill 的优先级修正

内容发布 skill 的第一层判断应该是用户问题和品牌心智，而不是平台矩阵、canonical、标签或格式检查。SEO/GEO 的目标也不是单纯提高 eunomia.dev 单站权重，而是提高整个品牌矩阵的权重：eunomia.dev 提供 canonical archive，GitHub 提供 artifact 和工程可信度，论文提供研究可信度，demo/talk 展示可体验证据，站外平台承载原生讨论和分发。

默认判断顺序：

1. **用户痛点**：目标读者现在遇到什么问题，为什么会主动搜索或参与讨论。
2. **搜索/社区意图**：他们会用什么关键词、在哪些平台提问、期待教程、对比、案例、数据还是工具。
3. **现有替代方案**：他们现在会用 SDK/OTel、MCP proxy、sandbox、Falco/Tetragon、商业安全平台、手工日志，还是只靠 prompt 和审批。
4. **我们的独特证据**：是否有公开 GitHub artifact、论文数据、benchmark、trace、demo、截图、issue 或真实工程边界。
5. **品牌心智**：内容主要服务 AI Agent Observability & Harness、eBPF Infrastructure、GPU & Systems Research，还是具体 paper/tutorial/release。
6. **平台执行**：长文默认 canonical syndication，Medium/Dev.to 发英文版，知乎/掘金发中文版；X/LinkedIn 发简短观点和 share link；Reddit/HN/Lobsters 只在问题和社区 fit 明确时选择性参与。

canonical、description、tag、OG image、schema、llms-full、平台预览和 ledger 都是发布卫生项。它们重要，但不应该反过来决定内容优先级，也不应该把一个不解决用户问题的内容包装成“值得发布”。

用户痛点的稳定版本单独维护在 `draft/user-pain-map.zh.md`。skill 可以引用这个文档里的稳定分类，但不应该把临时市场判断、大段战略叙事或未验证渠道假设复制进 skill。

## 内容操作系统

1. **信号收集层**：定期看 GitHub issues/releases/stars、kernel/eBPF 动态、论文、HN/Reddit/Lobsters、X/LinkedIn、知乎/掘金、Search Console、AI 搜索回答。
2. **选题判断层**：每个候选题记录来源、时效性、目标读者、独特角度、可用证据、适合平台、风险边界。
3. **高质量报告层**：每两周产出一篇技术 report 或趋势 synthesis，优先展示判断力，而不是宣传项目。
4. **canonical archive 层**：eunomia.dev 保存最完整、最可引用版本，覆盖 blog、tutorials、报告、项目页、llms-full、结构化数据和稳定 URL。
5. **平台原生分发层**：每个平台重写角度，而不是复制粘贴。平台帖先给读者价值，再给项目或原文链接。
6. **ledger 与复盘层**：一个平台一个 JSON，记录 source、发布状态、URL、截图、互动、后续动作和缺口。

默认 source set 是当前 `docs/blog/posts/` 加 `docs/tutorials/`，不再把 legacy 内容作为默认盘点对象。

## 选题原则

- 保持 80% contribution / 20% promotion：先解释问题、机制、数据、边界，再出现项目链接。
- 项目是证据，不是主角。项目只有在帮助读者复现、比较、检查源码或继续学习时才出现。
- 蹭热点必须有自己的技术角度：能给出 eBPF/runtime/agent/security/GPU 系统视角，才值得写。
- 不追平台算法玄学。每次策略调整要有可见证据：官方文档、浏览器可见平台规则、搜索结果、账户数据、GitHub 数据或真实反馈。
- 选题优先级按五项打分：品牌契合度、时效性、独特技术角度、证据质量、平台适配度。

## 节奏

默认每天都有一个可见产出。可见产出不等于每天都发一篇完整长文，而是每天至少交付一个能推进品牌运营的 artifact：站内草稿、平台草稿、已发布平台帖、issue/docs 回链、topic radar、ledger 更新、复盘报告或可截图的发布页面。

重内容保持分层节奏：每天一个小产出，每两天一个可分发 content unit，每两周一篇高质量 report 或趋势 synthesis，每月一次平台与选题复盘。可以轮换：

- 站内 blog/tutorial/report
- X/LinkedIn 短观点或 thread
- 知乎/掘金中文长解释
- Reddit/HN/Lobsters 的选择性讨论
- GitHub issue/docs 回链
- 月度或双周 report 摘要

下一阶段执行计划见 `draft/plan/README.zh.md` 和对应日期文件；`draft/content-output-plan-2026-07.zh.md` 只保留索引。

Medium/Dev.to 适合延迟转载，支持 canonical 时回指 eunomia.dev。知乎、掘金、X、LinkedIn、Reddit、HN、Lobsters 更重要的是平台原生可信度，不要把每篇都写成外链漏斗。

## 平台角色

- **eunomia.dev**：canonical archive、作品集、GEO source、长文和教程中心。
- **GitHub**：可信实现、issue 反馈、release 证明、代码入口。
- **X**：即时技术观点、线程、实验进展、会议/论文节点。
- **LinkedIn**：专业判断、研究/开源影响、工程领导力。
- **知乎**：中文系统解释、问题回答、研究判断。
- **掘金**：开发者教程、工程实践、代码和复现导向内容。
- **Reddit/HN/Lobsters**：只发真正有讨论价值的技术内容，少宣传，多问题意识。
- **Medium/Dev.to**：英文长尾转载和开发者发现渠道，注意 canonical。
- **小红书/B 站/YouTube**：等图卡或视频产能稳定后再系统投入。

## Skill 边界

skill 应该装：

- 明确触发条件和反触发条件。
- 可重复执行的步骤，例如读源文、改写、打开浏览器、粘贴、预览、停在发布前、截图、更新 ledger。
- 安全边界，例如平台发布必须走正常浏览器 UI，禁止直接访问隐藏 API 或平台内部 endpoint。
- 平台偏好和 QA 清单，例如标题、标签、链接、图片、alt text、canonical、评论跟进。
- 可验证脚本，例如 ledger 检查、frontmatter 检查、链接检查、schema/llms 校验。
- 少量稳定 heuristics，且要标明官方文档、浏览器可见状态和当前用户指令优先。

skill 不应该装：

- 长期品牌战略、商业计划、渠道优先级、内容日历总规划。
- 私有业务信息、客户/融资/定价/竞争情报。
- 平台算法传闻、未经验证的增长玄学。
- 需要频繁随市场变化重写的大段战略叙事。

## 更新机制

- 每天至少更新一个产出记录：topic radar、草稿状态、平台草稿、发布 URL、截图、ledger 或复盘 note。
- 每周做一次 backlog grooming：确认下周每天的产出对象、负责人、目标平台和停止点。
- 每两周产出一份高质量 report 或趋势 synthesis。
- 每月复盘平台数据、GitHub 反馈、AI/搜索引用、未发布缺口和下月选题。
- 当官方 SEO/GEO 或平台规则变化时，先更新 draft 里的判断；只有变成稳定、可执行、可验证的流程后，才进入对应 skill。
