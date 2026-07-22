---
date: 2026-07-20
slug: agent-work-unit
title: 从 Token 到可验证工作：Agent Infra 正在改写它的计量单位
description: 当 AI Agent 从一次模型调用变成长时间、多工具的执行系统，基础设施需要从 token 转向包含验收条件、执行证据、总成本和责任边界的可验证工作单元。
research_question: 当 Agent 从一次模型调用变成长时间、多工具的执行系统，基础设施应该以什么作为观测、评估和计费的基本单位？
research_window: 2026-07-18 至 2026-07-20，最近工作日回溯至 2026-07-17，机制与反证回溯 30 天
tags:
  - daily-analysis
  - research
  - AI Agent
  - Agent Infrastructure
  - Observability
---

# 从 Token 到可验证工作：Agent Infra 正在改写它的计量单位

一次模型调用容易记账：哪个模型、多少 token、多长延迟、有没有报错。Agent 打破了这种简洁。一次用户请求现在可能跨越几十次调用、多个工具、人工审批、重试、外部系统的状态变化。调用仍然可观测，但工作有没有真的完成，是另一个问题。

过去一周的一组发布让这个错位很难忽视。[OpenAI](https://openai.com/index/a-scorecard-for-the-ai-age/) 提议用完成的有用工作而非原始能力来衡量 AI。[Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/13-demos-on-gemini-enterprise-agent-platform) 演示 Agent Platform 的完整生命周期，包括构建、持续数周的运行、治理和评估。[GitHub](https://github.blog/changelog/2026-07-17-repository-level-github-copilot-usage-metrics-generally-available/) 开始按仓库和 PR 报告 Copilot 活动。阿里云的 [AgentLoop](https://www.alibabacloud.com/en/notice/commercialization_notice_for_agentloop_795?_p_lc=1) 于 7 月 20 日转入商用，把 AI Agent 的会话、工具调用、token 和 trace 打包成一个可计费的观测层。

产品线不同，方向相同。**Agent infra 正在从管理模型调用转向管理工作。行业已经能记录 Agent 做的大部分事情，但还没有共识：工作何时算完成，结果是否正确，一次可靠完成究竟花多少钱。**

<!-- more -->

## 四层，每一层都看不见上面那一层

[OpenTelemetry 的 GenAI conventions](https://opentelemetry.io/blog/2026/genai-observability/) 把 Agent 执行拆成多种 span，包括 `invoke_agent`、模型调用和 `execute_tool`，并附上 token 数、延迟和结果。这是纯遥测：系统做了什么、哪里慢、哪里坏。排查重试风暴或 token 跑飞，靠的就是这一层。

Agent 平台在其上加了一层：会话或执行轨迹。AgentLoop 把轨迹定义为从用户输入经模型、工具、检索、记忆到最终响应的完整路径。Google 的示例加入持久状态机、暂停恢复、人工审批。这一层比调用更接近真实执行，因为一个任务可以跨很多调用，也可以跑上好几天。

GitHub 新的仓库级 Copilot 指标再往上走。它统计 AI Agent 创建了多少 PR、合并了多少，以及 review 接受了多少条建议，计量的是交付活动而非推理请求。用来看 AI 在哪些仓库比较活跃，很方便；用来判断合并的改动是否正确、必要或真的降低了维护负担，就不太够了。

OpenAI 的 scorecard 瞄得最高：完成了多少有用工作、每个成功任务的完整成本、可靠性、规模化后每一美元的价值。成本明确包含重试、人工复核和返工，而不只是 token 单价。最接近业务结果，也最难自动度量。

叠起来：

`调用遥测 → 执行轨迹 → 交付活动 → 可验证结果`

每一层从更广的视角描述同一份工作，但不能互相替代。调用成功说明不了轨迹的连贯性，轨迹完整说明不了 PR 是否值得合并，PR 合并了也说明不了用户的问题是否解决。仪表盘越来越丰富，决策不会自动跟上。

## 为什么是现在

Agent 同时在变长、变宽、变脆。Google 的 13 个示例从简单 ADK agent 一直跨到能跑数周的持久工作流，带 MCP 工具、多 Agent 管道、身份联邦、人工审批门禁，还有一个由 OTel trace 喂养的评估飞轮。一个 prompt tweak 可能改善三个示例，也可能破坏另外十个。回归测试不再是可选项。

成本结构随形态改变。短对话可以按 token 计价，长任务不行：规划质量、工具失败、权限阻塞、重试、人工等待、返工都进账单。便宜模型如果要三次尝试，成本可能比贵模型一次通过还高。模型路由变成对预期完成成本的优化，而不是比较每 token 价格。

商业产品已经围绕这条链收费。AgentLoop 的文档把 trace 转数据集、评估、实验、prompt 版本管理、灰度发布和审计放在一个闭环里。AI Agent 接入指南现在附带本机 collector 和 hooks，可采集 Claude Code、Codex、Cursor 等工具的会话。值得注意的不是某一家产品是否兑现了所有承诺，而是产品边界已经从 LLM observability 扩展到了 Agent 的生产生命周期。

## 轨迹不是结果

矛盾就在这些发布里。GitHub 数 PR 的创建和合并，OpenAI 想从“采用了多少”转向“完成了什么”。Google 和阿里云都把自动评估放在观测之后，从生产 trace 生成数据集，再让 LLM judge 抓回归。大家都在向结果靠拢，但落点仍是不同强度的代理指标。

困难不是理论上的。本月初的一项 [benchmark 审计](https://arxiv.org/abs/2607.01211) 在四类云机器上重放 740 个性能优化任务。在一种机器上通过的参考补丁，在另一种上失败。排名也会随评分规则变化：28 对两两比较里有 9 对顺序随聚合方式翻转。研究针对的是性能 benchmark，不能否定所有 Agent 评估，但标出了一个真实的边界：即使有可执行测试和数值分数，环境和聚合仍可能翻转结论。

[DeepSWE](https://arxiv.org/abs/2607.07946) 指向了更好的方向。它为 91 个仓库中的 113 个长任务手写行为 verifier，在隔离环境里检查用户要求的功能是否可用，而不是检查补丁是否与参考相同。它同时发布 verifier 和完整轨迹。独立 LLM judge 与 DeepSWE verifier 的分歧率为 1.4%，与 SWE-Bench Pro 继承测试的分歧率为 32.4%。手写 verifier 不等于真实用户价值，但把任务描述、执行环境、verifier 和轨迹打包在一起，比沿用合并 PR 时附带的测试更接近可复核的“完成”。

漂亮的生命周期图不会消除基础设施故障。7 月 20 日，Google Developer Forums 上还有用户报告 [Gemini Enterprise 的自定义 MCP reload 返回 401](https://discuss.google.dev/t/gemini-enterprise-custom-mcp-reload-custom-actions-always-fails-with-401-ui-uses-api-key-instead-of-oauth-token/371907/2)。这是单个案例，不能代表平台整体，但说明为什么 trace 必须绑定外部状态。trace 可能完美记录了调用，用户需要的 action 却从未生效。

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
