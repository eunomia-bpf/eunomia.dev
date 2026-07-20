---
report_id: 2026-07-20-agent-work-unit
title: 从 Token 到可验证工作：Agent Infra 正在改写它的计量单位
research_question: 当 Agent 从一次模型调用变成长时间、多工具的执行系统，基础设施应该以什么作为观测、评估和计费的基本单位？
research_window: 2026-07-18 至 2026-07-20，最近工作日回溯至 2026-07-17，机制与反证回溯 30 天
source_cutoff: 2026-07-20 00:35 PDT
status: draft
thesis: Agent 平台正在把管理单位从 token 和模型调用上移到任务与执行轨迹，但会话数、PR 数、轨迹和自动评分仍只是工作的代理指标。下一层真正缺失的基础设施原语，是同时包含验收条件、执行证据、总成本和责任边界的可验证工作单元。
---

# 从 Token 到可验证工作：Agent Infra 正在改写它的计量单位

过去几年，AI 基础设施最容易回答的是一次模型调用发生了什么。用了哪个模型，输入和输出多少 token，延迟多长，是否报错。Agent 把这个问题拉长了。一次用户请求现在可能跨越几十次模型调用、多个工具、长时间暂停、人工审批、失败重试和外部系统状态变化。单次调用仍然可观测，却越来越难说明工作是否真的完成。

最近一轮产品与工程信号把这种错位推到了台前。7 月 17 日，[OpenAI](https://openai.com/index/a-scorecard-for-the-ai-age/) 提议用完成的有用工作、成功任务成本和结果可靠性衡量 AI。[Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/13-demos-on-gemini-enterprise-agent-platform) 把 Agent Platform 展示为从构建、持久运行、治理到评估的完整生命周期。[GitHub](https://github.blog/changelog/2026-07-17-repository-level-github-copilot-usage-metrics-generally-available/) 将 Copilot 指标下沉到仓库和 Pull Request 活动。阿里云则在 7 月 20 日开始对 [AgentLoop](https://www.alibabacloud.com/en/notice/commercialization_notice_for_agentloop_795?_p_lc=1) 商业计费，并把 coding agent 的会话、工具调用、token、日志和 trace 纳入同一个观测系统。

这些发布来自不同产品线，却指向同一个变化。**Agent infra 正在从“模型调用管理”走向“工作管理”。行业已经能记录一次工作的大部分过程，却还没有共同定义什么算完成、完成得是否正确，以及一次可靠完成究竟花了多少钱。**

## 四个同时出现、却不能混用的计量单位

[OpenTelemetry 的 GenAI semantic conventions](https://opentelemetry.io/blog/2026/genai-observability/) 已经能把一次 Agent 执行拆成 `invoke_agent`、模型调用和 `execute_tool` 等 span，并记录模型、token、延迟和工具结果。它处理的是遥测层问题，包括系统做过什么、哪里慢、哪里失败。对于排查重试循环、慢工具和异常 token 消耗，这一层不可替代。

Agent 平台正在其上增加第二层，即会话或执行轨迹。阿里云 AgentLoop 的文档把完整轨迹定义为从用户输入开始，经过模型、工具、检索和记忆，直到最终响应的全过程。Google 的 Agent Platform 示例则加入持久状态机、暂停与恢复、人工审批和跨 Agent 协议。这个单位比模型调用更接近真实执行，因为一个任务可以跨越多个调用，也可以运行数天。

GitHub 7 月 17 日上线的仓库级 Copilot 指标又往上走了一层。新接口按仓库报告 coding agent 创建和合并的 PR，以及 code review 处理的 PR 和建议数量。这里的计量对象已经不是推理请求，而是代码库里的交付活动。它能告诉管理者 AI 在哪些仓库产生了可见活动，却不能单独证明合并后的改动正确、必要或降低了维护成本。

OpenAI 同日提出的 scorecard 再向上一步，要求衡量完成了多少有用工作、每个成功任务的完整成本、结果是否可靠，以及每一美元能否随着规模扩大产生更多价值。它明确把重试、人工复核和返工放进成本，而不是只比较 token 单价。这个单位最接近业务结果，也最难自动获得。

把四层放在一起，可以得到一条逐步接近结果的链。

`调用遥测 -> 执行轨迹 -> 交付活动 -> 可验证结果`

它们描述的是同一次工作，却不是可互换的指标。调用成功不等于轨迹合理，轨迹完整不等于 PR 值得合并，PR 已合并也不等于用户问题得到解决。把较低层的可见数字直接当成较高层的结果，会让仪表盘越来越丰富，决策却没有同步变可靠。

## 为什么这个变化在现在发生

Agent 的工程形态正在同时拉长时间、扩大权限并增加失败路径。Google 的 13 个示例从简单 ADK Agent 一直覆盖到可运行数周的持久工作流、MCP 工具、A2A 多 Agent 管道、身份与网关、人工审批以及基于 OTel 数据的评估飞轮。一个 prompt tweak 可能改善三个样例，也可能破坏另外十个，因此部署动作自然需要历史数据、回归评估和失败聚类。

成本结构也随之改变。短对话可以近似用 token 计价，长任务却会受到规划质量、工具错误、权限阻塞、模型重试、人工等待和返工影响。便宜模型如果需要多次尝试，成功结果的总成本可能更高。昂贵模型如果一次通过验收，反而可能更便宜。模型路由由此不再只是每 token 价格比较，而是对“成功概率乘以完整执行成本”的优化。

商业产品已经开始围绕这条链收费和竞争。AgentLoop 在 7 月 20 日进入按量收费，其文档把 trace 转数据集、评估、实验、prompt 与 skill 版本管理、灰度发布和审计放进同一闭环。7 月 17 日更新的 coding agent 接入文档还提供本机 collector，用 hooks 或 plugins 采集 Claude Code、Codex、Cursor 等工具的会话与调用数据。这里值得注意的不是某一家产品是否已经兑现全部宣称，而是产品边界已经从 LLM observability 扩展到了 Agent 的生产生命周期。

## 轨迹不是结果，自动评分也不是验收

最明显的矛盾来自这些发布本身。GitHub 的仓库级指标可以数出 PR 的创建、合并和 review 建议，OpenAI 却主张从“采用了多少”转向“完成了什么”。Google 和阿里云都把自动评估放在观测之后，希望从线上 trace 生成数据集，再由 AutoRater、LLM-as-a-Judge 或 Agent-as-a-Judge 识别回归。大家都在向结果靠近，但实际落点仍是不同强度的代理指标。

自动评估的困难不是理论上的担忧。7 月初的一项 [coding agent benchmark 审计](https://arxiv.org/abs/2607.01211) 在四类云机器上重放 740 个性能优化任务，发现多个 benchmark 的参考补丁在跨机器环境中并不稳定。共同提交的排名还会随评分规则变化，28 对比较里有 9 对次序不一致。这个结果针对性能优化 benchmark，不足以否定所有 Agent 评估，却清楚说明了它的边界。即使有可执行测试和数值分数，环境、oracle 和聚合方式仍可能改变结论。

[DeepSWE](https://arxiv.org/abs/2607.07946) 提供了一个更接近“可验证工作单元”的正面对照。它为 91 个代码仓库中的 113 个长任务编写行为 verifier，在隔离环境里检查用户要求的功能，同时允许与参考补丁不同的实现。论文还释放 verifier 和完整轨迹。独立 LLM judge 与 DeepSWE verifier 的结论分歧率为 1.4%，与 SWE-Bench Pro 继承测试的分歧率则为 32.4%。这不能证明手写 verifier 等同于真实用户价值，却表明“任务描述、执行环境、验收器和完整轨迹”放在一起，比沿用仓库里原有测试更接近可复核的完成定义。

社区层面的摩擦也提醒我们，漂亮的生命周期图不会消除基础设施故障。Google Developer Forums 在 7 月 20 日仍列出用户对 [Gemini Enterprise 自定义 MCP reload 返回 401](https://discuss.google.dev/t/gemini-enterprise-custom-mcp-reload-custom-actions-always-fails-with-401-ui-uses-api-key-instead-of-oauth-token/371907/2) 的报告。这只是个案，不能代表平台整体可靠性，但它展示了任务结果为何必须绑定外部状态。trace 可能完整记录了调用，用户需要的 action 却没有真正生效。

因此，Agent 系统需要的不只是更多 trace，也不是给每条 trace 再附一个总分。更稳健的抽象应该把一次工作定义为可验证工作单元，至少包含：

- 明确的意图、目标对象和“完成”条件。
- 执行时适用的权限、策略和人工审批边界。
- 模型调用、工具调用、状态变化和产出 artifact 的证据链。
- 能确定性检查的 outcome oracle，例如测试、部署状态、账本变化或目标系统回执。
- 无法确定性判断时的评分依据、置信度、反例和人工复核结果。
- token、运行时间、重试、人工等待与返工构成的总成本。
- Agent、模型、prompt、skill、工具和环境版本，以便重放与归责。

[AgentSight](https://arxiv.org/abs/2508.02736) 在 Agent 与系统边界关联语义意图和内核可见效果，论文报告的额外开销低于 3%，并展示了对 prompt injection、推理循环和多 Agent 瓶颈的识别。它补的是执行证据，不是业务结果 oracle。一条因果轨迹可以证明 Agent 做过哪些系统动作，却不能独自证明用户目标已经实现。

[ActPlane](https://arxiv.org/abs/2606.25189) 补的是权限与策略上下文。其对 1,127 条系统可观测策略的研究显示，73.6% 需要项目或任务上下文，跨事件策略中这一比例达到 95%。这意味着验收时不能只回看 Agent 调了什么工具，还要知道当时解析出了什么策略、获得了什么授权、执行环境应用了哪个版本。AgentSight 和 ActPlane 分别提供证据链与责任边界，但都没有替代 outcome oracle、总成本和人工复核。把它们放进报告，是为了标出完整工作单元仍缺哪些字段，而不是把现有项目包装成完整答案。

这样的工作单元不会替代 OTel span 或平台 trace。它把低层遥测和高层验收连接起来，让系统能够回答两个不同的问题。为什么这次执行变慢了，以及它最后有没有把正确的事情做完。

## 二阶影响：基础设施会从看见 Agent 变成管理改变

一旦结果而非调用成为优化目标，观测数据就不再只是故障现场。线上失败会进入数据集，数据集成为回归测试，回归结果决定 prompt、skill、模型或工具版本能否发布。Google 所说的 evaluation flywheel 和 AgentLoop 的 Trace2Dataset 都在走这条路。真正有价值的飞轮并不是数据越多越好，而是失败能否被转成稳定、可重放且能阻止同类回归的验收条件。

prompt 和 skill 也会越来越像代码资产。它们需要版本、评审、灰度和回滚，因为 Agent 行为的变化可能来自模型，也可能来自 instruction、工具权限或环境准备步骤。只记录最终模型版本会漏掉大量因果变量。

可观测性与隐私之间的张力还会加剧。OpenTelemetry 默认不采集 prompt 内容和工具参数，因为这些字段可能包含敏感数据。AgentLoop 的本机采集文档也专门提供凭据遮罩配置。结果验证往往需要更多上下文，但记录越完整，泄露面和合规成本越大。未来的可验证工作单元必须允许证据最小化，例如只保存测试摘要、哈希、签名回执或受控引用，而不是默认保存完整思维链和业务数据。

## 反面观点与不确定性

这组信号主要来自厂商文档和产品发布，能够证明产品方向，不能证明企业已经普遍采用，也不能证明这些闭环能稳定提高 Agent 质量。AgentLoop 关于诊断时间、异常成本和可追溯性的数字属于供应商陈述，仍需要独立案例或公开实验验证。

“定义完成”也不是所有任务都同样容易。编译、测试、部署和账本状态可以形成较强 oracle，研究、设计、沟通与探索性工作则需要多维判断。强行给后一类任务压成单一分数，可能奖励容易度量的浅层产出，并抑制有价值但不确定的探索。

这篇报告的 thesis 会被两类证据推翻。第一，如果后续独立部署显示调用级或会话级指标已经足以稳定预测真实业务结果，那么增加工作单元只会制造复杂度。第二，如果跨模型、跨 Agent、跨平台的结果定义无法迁移，所谓统一原语可能只能退化为每个工作流的定制 schema。目前证据更支持“需要一层连接机制”，还不足以证明最终标准会长成什么样。

## 开发者现在可以做什么

不必等待平台给出统一答案。先选择一个重复发生、结果可检查的 Agent 工作流，把“完成”写成外部系统可以验证的状态，而不是“Agent 返回成功”。代码任务可以要求测试通过且目标 diff 存在，发布任务可以要求公开 URL、内容校验和 ledger 一致，运维任务可以要求资源状态与变更回执同时满足。

随后把活动指标和结果指标分开。token、延迟、工具错误和 trace 用来解释执行，成功率、回归率、人工复核、重试次数和每次成功成本用来判断价值。不要把 PR 数、会话数或 judge 总分放进一个看似统一的“生产力”数字。

最后，为失败建立可重放入口。保存最小必要的环境版本、输入条件、artifact 和验收结果，把高价值失败加入回归集。确定性 oracle 优先，语义 judge 负责无法编码的部分，并对低置信度或高风险任务保留人工门槛。这样，观测才会真正进入改进闭环，而不是停在仪表盘上。

## 还没有回答的问题

- 是否能定义一套可移植 schema，把任务目标、系统效果、验收结果、版本和完整成本连接起来，而不强迫所有工作流共享同一个 outcome oracle。
- PR、会话和轨迹等活动指标，在哪些条件下能预测用户可见结果，哪些条件下会系统性失真。
- verifier 如何抵抗机器类型、依赖版本、外部服务和数据状态的漂移，并在环境变化后保持可重放。
- 自动 judge 如何被校准、审计和版本化，尤其是 evaluator 本身也会调用工具和执行多步推理时。
- 证据链如何做到最小披露，使验收、归责和隐私保护能够同时成立。
- 哪些工作单元字段可以跨 Agent 与平台复用，哪些必须由具体业务工作流定义。

## 参考资料

- [OpenAI, A scorecard for the AI age, 2026-07-17](https://openai.com/index/a-scorecard-for-the-ai-age/)
- [Google Cloud, 13 hands-on demos to build on Gemini Enterprise Agent Platform, 2026-07-17](https://cloud.google.com/blog/products/ai-machine-learning/13-demos-on-gemini-enterprise-agent-platform)
- [GitHub Changelog, Repository-level GitHub Copilot usage metrics generally available, 2026-07-17](https://github.blog/changelog/2026-07-17-repository-level-github-copilot-usage-metrics-generally-available/)
- [Alibaba Cloud, What is AgentLoop, updated 2026-06-22](https://www.alibabacloud.com/help/en/cms/cloudmonitor-2-0/what-is-an-agentloop)
- [Alibaba Cloud, Onboarding AI Coding Agents, updated 2026-07-17](https://www.alibabacloud.com/help/en/cms/ai-application-access-ai-coding-agent)
- [OpenTelemetry, Inside the LLM Call: GenAI Observability with OpenTelemetry, 2026-05-14](https://opentelemetry.io/blog/2026/genai-observability/)
- [DeepSWE: Measuring Frontier Coding Agents on Original, Long-Horizon Engineering Tasks, 2026-07-08](https://arxiv.org/abs/2607.07946)
- [Chen et al., Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?, arXiv v2, 2026-07-16](https://arxiv.org/abs/2607.01211)
- [AgentSight: System-Level Observability for AI Agents Using eBPF, arXiv v2, 2025-08-15](https://arxiv.org/abs/2508.02736)
- [ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses, arXiv v2, 2026-06-30](https://arxiv.org/abs/2606.25189)
