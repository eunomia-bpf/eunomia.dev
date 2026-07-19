---
date: 2026-05-25
slug: runtime-security-for-ai-agents
description: AI 编程 Agent 在平台方可能并不拥有的 Harness 与沙箱中自主运行数小时，基于审批的管控随之失效。本文主张将 Agent 安全拆分为三层（意图授权、执行隔离、副作用验证），并用基于 eBPF 的可观测（AgentSight）与执行控制（ActPlane）在 Harness 之下提供独立的运行时可观测与执行控制。
---

# 基于 eBPF 的不透明 AI Agent 运行时可观测与执行控制：超越沙箱与审批

AI 编程 Agent 如今可以连续运行数小时，端到端地完成整个功能开发，优化生产级
GPU 内核，并自主合并数千个 Pull Request。然而，大多数 Agent 安全仍然依赖
人工审批（human-in-the-loop），而 Anthropic 自己的数据显示用户在 93% 的权限
提示上直接点了同意，缺乏实质性审查。结果可以预见：产品纷纷添加绕过模式，
用户关闭权限门控，65% 的企业报告了 Agent 安全事件。

但更深层的问题并非审批疲劳。真正的问题在于：Agent harness（提示循环、工具
路由、权限逻辑、沙箱默认配置）正日益成为平台团队未曾编写的第三方产品，
运行在平台团队可能并不拥有的沙箱中。Harness 不是可信的安全边界。本文主张
将 Agent 安全拆分为三个层次、三个不同的责任方：意图授权（harness 拥有）、
执行隔离（所有权存在争议）、副作用验证（必须由平台方拥有）。当三层一致时，
你可以放心。当它们不一致时，你需要在操作系统层面进行独立的可观测和执行控制
来发现分歧，而这恰恰是大多数 Agent 平台缺失的一层。我们正在朝这个方向构建
两个开源项目：[AgentSight](https://github.com/eunomia-bpf/agentsight/) 用于
运行时观测，[ActPlane](https://github.com/eunomia-bpf/ActPlane) 用于运行时
harness 执行控制，两者都使用 eBPF 在 Agent harness 之下提供独立的运行时可观测与执行控制。

<!-- more -->

## 为什么是现在：能力飙升，护栏落后

2026 年的重要变化不是 Agent 的出现，而是它们所做事情的规模和持续时间。

一年前，典型的 Agent 任务是"修复这个 bug"或"写这个函数"。到了 2026 年，
Agent 常规性地在复杂的多步骤工作中连续运行数小时。OpenAI 记录了一个 Codex
会话[连续运行 25 小时](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex)，
消耗 1300 万 token，从空仓库生成了 3 万行代码。Anthropic 的 Agentic 编程
报告引用了[一次 7 小时内完成的 1250 万行代码库变更](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf)。
Meta 的 [KernelEvolve](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/)
使用多 Agent 协调来编写和优化生产级 GPU 内核，将此前需要数周专家系统工程的
工作压缩到几小时内。在 SWE-bench Verified 上，[顶级 Agent 现在能解决
60-70%](https://www.vals.ai/benchmarks/swebench) 的真实 GitHub Issue，而
2024 年初不到 30%。Devin 已在企业客户中[合并了数十万个 Pull Request](https://cognition.ai/blog/devin-annual-performance-review-2025)，
合并率达 67%。Goldman Sachs 在 12,000 人的工程团队中[部署了数百个 Devin
实例](https://www.cnbc.com/2025/07/11/goldman-sachs-autonomous-coder-pilot-marks-major-ai-milestone.html)。

不只是编程 Agent，通用自主 Agent 也已进入主流。
[OpenClaw](https://github.com/openclaw/openclaw) 是一个拥有超过 30 万
GitHub star 的开源 Agent，可连接 LLM 并在用户机器上执行 shell 命令、浏览器
自动化、邮件、日历和文件操作。CrowdStrike 称其为[安全团队需要关注的"AI
超级 Agent"](https://www.crowdstrike.com/en-us/blog/what-security-teams-need-to-know-about-openclaw-ai-super-agent/)：
2026 年 1 月至 4 月期间，在三轮披露中针对它提交了 [470 个安全公告](https://www.reco.ai/blog/openclaw-the-ai-agent-security-crisis-unfolding-right-now)。

这些不是研究演示。它们是生产工作流：后台任务、并行执行、多小时会话、端到端
功能开发、内核优化、企业级代码变更。

**与此同时，旨在保障 Agent 安全的护栏并没有跟上。**

大多数 Agent 安全仍然依赖人工审批：一个提示要求用户在每个操作执行前批准或
拒绝。这对只有几次工具调用的短会话有效，但在 Agent 在数小时的自主操作中做出
数百个决策时就行不通了。

证据表明，基于审批的控制在实践中已经失效。Anthropic 自己的数据显示，[Claude
Code 用户在 93% 的权限提示上直接同意](https://www.anthropic.com/engineering/claude-code-auto-mode)，
这一比率与盲目点击一致，而非有意义的审查。一项对 Claude Code 自动模式的
独立压力测试发现，在模糊的状态改变操作上有 [81% 的漏报率](https://arxiv.org/abs/2604.04978)，
意味着分类器放行了 5 个应需人工审查操作中的 4 个。真实事件随之而来：在有
记录的案例中，关闭权限门控运行 Agent 的用户，其[主目录被 Agent 生成的
`rm -rf` 命令删除](https://gist.github.com/hartphoenix/698eb8ef8b08ad2ce6a99cf7346cd7cc)。
2026 年的一项行业调查发现，[65% 的企业报告了 AI Agent 安全事件](https://www.kiteworks.com/cybersecurity-risk-management/ai-agent-security-incidents-2026/)，主要
涉及未授权数据访问、凭证泄露和向外部端点的数据外泄，其中大多数涉及缺乏适当
Agent 访问控制的组织。

产品通过添加绕过机制来应对。Claude Code 提供了
`--dangerously-skip-permissions`。Windsurf 的 Cascade Agent [自主执行](https://stackbuilt.co/blog/windsurf-vs-cursor-2026)，
而 Cursor 则会停下来询问。社区指南现在聚焦于"如何安全使用 YOLO 模式"。
Anthropic 研究员 Nicholas Carlini [同时运行了 16 个绕过权限的 Claude
Agent](https://x.com/nicholas_carlini)，附带警告："在容器里跑，别在你的真机上跑。"

这就是矛盾所在：**Agent 越强大，用户越想让它们不受打扰地运行，而人工审批作为
主要安全边界就越无效。**

这一矛盾正是催生不同安全模型的原因。

## 问责缺口

更深层的问题不仅仅是 Agent 更强大了，而是 Agent harness（决定 Agent 行为的
组件）正日益成为平台团队未曾编写的第三方产品。

现代 Agent harness 不是模型的薄封装。它包括提示循环、规划和重试逻辑、工具
路由、MCP 客户端、权限模式、审批门控、hooks、记忆、日志、凭证处理，有时还
包括沙箱默认配置。在许多部署中，这个 harness 来自托管的编程 Agent 服务或
平台团队无法控制的开源框架。

这在整个生态系统中已经清晰可见。GitHub Copilot 的[编程 Agent](https://docs.github.com/en/copilot/concepts/about-copilot-coding-agent)
在 GitHub Actions 中自主运行，研究代码库、制定计划、进行变更、打开 Pull
Request。OpenAI [Codex](https://developers.openai.com/codex/cloud) 在沙箱化
的云环境中运行后台任务，具有受控的网络访问。Claude Code 在 Anthropic 管理
的 VM 中运行云会话，使用有范围限制的凭证。Kubernetes SIG 正在定义
[Agent Sandbox](https://agent-sandbox.sigs.k8s.io/) 用于隔离的、有状态的
Agent 工作负载。最近的研究数据集展示了真实代码库中[大规模的 Agent 编写的
Pull Request](https://arxiv.org/abs/2602.09185)。

所有权的分离现在在主要平台中已经明确。Anthropic 的共担责任框架[将 Agent
安全划分为四层](https://www.anthropic.com/research/trustworthy-agents)（模型、Harness、工具、环境），
并强调 Agent 的行为取决于这四层协同工作，因此由部署方塑造的 Harness、工具和环境
与模型本身同样关键。Anthropic 同时也承认，即使这些分层防护加在一起也"不是
保证"。该框架留下的问题是：当某一层发生失败时，部署方是否有独立的可观测来发现
它。在云基础设施中，共担责任模型中类似的缺口（客户"拥有"配置但无法观测运行时
行为）催生了 CloudTrail、Config、GuardDuty 等由客户控制的独立可观测与审计服务。Agent
基础设施还没有等价物：部署方被告知它拥有 harness、工具和环境的责任，但通常
没有独立的方式来验证这些层在运行时到底做了什么。

GitHub 的 Agentic 工作流架构从这一前提出发：
["Agent 默认不可信，尤其在存在不可信输入的情况下"](https://github.blog/ai-and-ml/generative-ai/under-the-hood-security-architecture-of-github-agentic-workflows/)，
使用内核级通信边界，即使 Agent 容器被攻破也能保持有效。OpenAI 的 Codex
文档[承认](https://developers.openai.com/codex/agent-approvals-security)
"devcontainer 提供了实质性保护，但无法防止所有攻击。"

平台团队仍然拥有代码库、CI Runner、Kubernetes 集群、服务账号、密钥和内部
网络。但作用于这些资产的运行时可能是不透明的。

还有第二个分离对平台团队更为重要：**沙箱也可能不受环境拥有者控制。** 如果
Agent 运行在提供商管理的云中（Web 版 Claude Code 运行在 [Anthropic 管理的
隔离 VM](https://docs.anthropic.com/en/docs/claude-code/security) 中，具有
有范围限制的凭证代理；Codex 运行在 [OpenAI 管理的容器](https://developers.openai.com/codex/concepts/sandboxing)中），
平台团队无法挂载自己的监控、修改隔离策略或检查沙箱内部。即使 Anthropic 自己
的托管 Agent 架构也明确地[将"大脑"（Claude + harness）与"双手"（沙箱）
解耦](https://www.anthropic.com/engineering/managed-agents)，将容器视为可丢弃的，确保 token 永远不会从运行生成代码的沙箱中可达。
这是好的架构，但它是提供商的架构，不是平台团队的。

当 Agent 在本地或自托管基础设施上运行时（GitHub 现在[支持自托管
Runner](https://github.blog/changelog/2025-10-28-copilot-coding-agent-now-supports-self-hosted-runners/)
用于其编程 Agent，Kubernetes Agent Sandbox 在平台运营者控制下提供
[gVisor/Kata 支持的隔离](https://agent-sandbox.sigs.k8s.io/)），环境拥有者
可以在 Agent 外面包裹自己的沙箱和可观测层。当 Agent 运行在提供商管理的环境中时，独立的可观测与执行控制
必须转移到平台团队确实控制的边界上。

这就产生了问责缺口：**平台团队要为一个无法完全审查的工作负载的生产影响负责，
而该工作负载运行在一个可能不属于自己的沙箱中。**

旧的心智模型很简单：Agent 有风险，把它放进沙箱。新的现实有着不同的信任边界：
Agent 及其 harness 是工作负载的一部分，环境拥有者需要独立的运行时可观测。

## 三层架构，三个问题

MCP、沙箱和操作系统级可观测对 Agent 安全都是必要的。它们不可互换。每一层
回答一个根本不同的问题，每一层有不同的责任方。

**意图授权**（MCP、工具网关、审批提示）回答：Agent *应该*做什么？它可以调用
哪些工具，以什么身份，具有哪些权限范围？这是在危险操作发生前实施访问控制的
正确位置。但工具审批不等于副作用的证明。框架日志显示"运行测试"并不能证明
进程树只运行了测试。一个 MCP 服务器可以身份验证完善，但仍然是导致意外本地
影响的工作流的一部分。这一层通常由 Agent harness 拥有或调解。

**执行隔离**（容器、VM、网络策略、命名空间）回答：Agent *能*触及什么？
哪些文件、网络端点、凭证和系统调用是可用的？这是限制爆炸半径的正确位置。
但沙箱不会自动记录 Agent 在其约束范围内尝试了什么：哪个进程读取了密钥，
哪个子进程打开了网络连接，沙箱策略是否与已批准的意图匹配。这一层的所有权
存在争议：它可能属于 Agent 提供商、平台团队，或两者都有。

**副作用验证**（OS/运行时可观测）回答：*实际*发生了什么？运行了哪些进程，
读取了哪些文件，打开了哪些网络连接，访问了哪些凭证？这一层提供关于执行的
事实，独立于框架报告或沙箱意图。这一层必须由环境运营者拥有。否则就不存在
独立的事实来源。

安全模型是三者的组合：

```text
授权意图          →  隔离执行            →  验证副作用
（harness 拥有）     （所有权存在争议）     （必须由平台方拥有）
```

当三层都一致时，你可以放心。当它们不一致时，你需要独立于 harness 的操作系统
级可观测和控制来发现不匹配、遏制损害并还原事件经过。

## 为什么独立性至关重要

保持这些层独立的原因源自上述趋势，但也来自关于所有权和信任的更深层结构性
论点。

### 审批疲劳

当审批在实践中被放松时（如上文所述），其他两层必须进行补偿。如果你自动批准
常规操作，你需要独立的方式来验证这些操作实际做了什么。如果你为了速度绕过
权限，你需要更强的隔离和更强的可观测。

### Harness 不透明性

当 harness 不透明时，应用层遥测不能作为唯一的事实来源。OpenTelemetry GenAI
规范和框架级追踪在你拥有框架时是有价值的。但不透明的 Agent 应用、闭源运行时、
托管执行、stripped 二进制文件以及任意的子进程树，都可能打破"框架追踪是完整的"
这一假设。OpenClaw 直接说明了这一点：其行为在[多次运行中是不确定的](https://arxiv.org/html/2603.27517v2)，
对相同输入产生不同的工具调用序列，这使得静态代码审查不足以应对，也驱使多个
团队为其构建了专门的运行时可观测工具
（[OneClaw](https://www.sentinelone.com/blog/oneclaw-discovery-and-observability-for-the-agentic-era/)、
[ClawTrace](https://www.epsilla.com/blogs/clawtrace-launch-openclaw-agent-observability)）。
安全研究人员已经在[所有主要 AI IDE 中发现了 30 多个漏洞](https://thehackernews.com/2025/12/researchers-uncover-30-flaws-in-ai.html)
（Cursor、Copilot、Windsurf、Claude Code），通过对 Agent 工具链的提示注入
实现数据窃取和远程代码执行。

MCP 层记录的是预期的工具调用。OS 层记录的是实际的副作用。当 harness 不透明
时，这两者之间的差距正是安全事件发生的位置。

### 信任边界就是所有权边界

独立性的最深层原因在于三层服务于不同的所有者，而这些所有者有不同的激励。

Harness 提供商的目标是完成用户的任务：最大化自主编程生产力、减少权限摩擦、
交付结果。平台团队的目标是保护代码库、密钥、集群、CI Runner、内部网络和
生产 API。这些目标并不对立，但也不完全相同。当它们冲突时，当完成任务的
最快路径涉及读取凭证、打开网络连接或修改工作区外的文件时，harness 会优化
任务完成，除非有独立的边界阻止它。

这就是为什么 [Bhattarai 和 Vu 认为](https://arxiv.org/abs/2602.09947)
"概率性合规不是合规"：基于训练和分类器的防御可能降低经验攻击率，但无法在
对抗条件下提供确定性保证。只有架构级的执行控制可以。Red Hat 在 Kagenti 上
部署多 Agent 系统的经验以不同的方式表达了同样的洞察：这是["一个伪装成 AI
问题的多租户问题"](https://next.redhat.com/2026/03/05/zero-trust-ai-agents-on-kubernetes-what-i-learned-deploying-multi-agent-systems-on-kagenti/)。
Agent 是不可信的租户。平台需要对其施加与任何不可信工作负载相同的隔离、身份
和审计控制。

[OWASP Agentic 应用 Top 10](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
强化了这一框架。其首要风险（ASI01，Agent 目标劫持）是"Agent 无法可靠地区分
指令和数据"，来自代码库、Issue、MCP 响应或网页的单个恶意输入就可以引导
Agent 使用其合法工具执行有害操作。这不是假设：[Bishop Fox 演示了](https://bishopfox.com/blog/otto-support-confused-deputy)
Confused Deputy 攻击，嵌入在支持工单中的指令导致 Agent 使用已授权的工具
窃取数据，"审计日志上的每条记录都是用户的名字。"[Docker 记录了](https://www.docker.com/blog/mcp-horror-stories-github-prompt-injection/)
一个 GitHub 提示注入链，一个恶意 Issue 劫持了 MCP 连接的 Agent，从私有
代码库中窃取机密数据。

因此，平台团队的威胁模型包含三个对手类别：

| 威胁 | 哪一层失效 | 运行时可观测检测到什么 |
|------|-----------|-----------------|
| **被劫持的 Agent**（提示注入、恶意代码库/Issue/MCP 响应） | 意图层：Agent 被诱导执行非预期操作 | 实际副作用偏离了声明的意图 |
| **不可信的 harness**（不透明的权限逻辑、不完整的日志、不可审计的内部状态） | 无法验证 harness 的完整性 | 独立于 harness 报告的 OS 级事实 |
| **沙箱逃逸或策略缺口**（容器逃逸、挂载的凭证、网络绕过） | 隔离层失效或配置错误 | 检测到预期沙箱边界之外的行为 |

AISI 的 [SandboxEscapeBench](https://arxiv.org/abs/2603.02277) 使第三类
威胁具体化：前沿模型可以在真实系统中可能出现的错误配置下可靠地逃逸容器沙箱，
研究者发现了四条基准设计者遗漏的意外逃逸路径。他们的建议：
["默认将普通 Docker 隔离视为不充分的。"](https://arxiv.org/abs/2603.02277)

在所有三种情况下，OS/运行时可观测都是让平台团队发现问题的独立控制手段，
无论哪一层失效。

## OS 级可观测采集什么

在 OS/运行时层，可观测采集的数据包括：

- **进程谱系**：从 Agent 到子进程到网络调用的完整树
- **文件访问**：读取或写入了哪些路径，包括凭证路径
- **网络行为**：连接、目标、时序、数据量
- **容器元数据**：命名空间、cgroup、Pod 身份、服务账号
- **子进程行为**：绕过框架插桩的命令

这些数据在应用层之下采集，通常通过 eBPF、审计子系统或内核插桩。它不需要
修改 Agent 应用。其关键特性是独立性：可观测由环境运营者拥有和运营，而非
Agent 提供商。

这使得跨层对比成为可能：

```text
框架报告：        运行测试
沙箱策略：        工作区已挂载，镜像仓库已允许，SA token 已挂载
OS 可观测：         agent → shell → python → curl
                  read: /var/run/secrets/.../token
                  connect: 未知外部主机
```

每一层看到了事件的不同部分。没有 OS 级可观测，这就是一次未被检测到的凭证窃取：
服务账号 token 被读取并发往外部主机，而框架日志只记录了"运行测试"。平台团队
数天后才发现泄露，如果能发现的话。OS 级可观测将一次隐形的数据泄漏变成实时检测。

## 部署现实

OS 级可观测在你控制 Agent 执行所在的主机、节点或 VM 时最为强大。如果 Agent
完全运行在提供商管理的环境中，你可能无法在其中挂载 eBPF。

在这种情况下，同样的模型适用，但可观测转移到你确实控制的边界上：

- 代码库权限和分支保护
- 最短生命周期的有范围限制的凭证
- CI/CD 和 GitHub 审计日志
- 网络代理和 webhook 事件
- 制品访问日志
- 提供商提供的会话日志

这些可观测比拥有运行时边界要弱，但仍然好过将 Agent 的对话记录作为唯一的
事实来源。

对平台团队来说，设计问题是：

> 我实际控制的最低层在哪里？
> 独立的可观测就应该建在那里。

## AgentSight 与 ActPlane：先观测，再执行控制

我们正在构建开源工具来实现上述验证层，分别解决问题的两个方面。

**[AgentSight](https://github.com/eunomia-bpf/agentsight/)** 是一个零侵入的
AI Agent 可观测工具。它使用 eBPF 在系统边界拦截 SSL/TLS 流量并监控进程行为，
不需要修改代码、不需要 SDK、不需要框架集成。将它指向任何 Agent 进程（Claude
Code、Codex、自定义 Python Agent），它就能捕获完整画面：进程谱系、LLM API
调用（提示和补全）、文件访问、网络连接和工具调用，全部关联到实时时间线中。
这就是"看到实际发生了什么"的层。因为它在应用层之下运行，即使 Agent 运行时
不透明、闭源，或运行任意绕过框架级追踪的子进程，它也能正常工作。在实践中，
这意味着在凭证访问、数据外泄企图和未授权网络连接发生时即刻检测，而不是数天后
由外部方报告泄露时才知道。

**[ActPlane](https://github.com/eunomia-bpf/ActPlane)** 是 AI Agent 的 OS
级 harness。AgentSight 负责观测，ActPlane 负责执行控制。你用基于 YAML 的
规则语言（标注信息流控制，而非静态允许列表）编写行为合约，ActPlane 将其编译
为 eBPF 程序，在内核级别执行约束：Agent 整个进程树中的每个 `exec`、文件打开
和网络连接都会对照策略进行检查。当规则被违反时，ActPlane 阻止该操作并通过
hook 系统将人类可读的原因反馈给 Agent，使 Agent 自我修正而非静默失败。规则
语言支持跨 fork/exec 链的数据流追踪、因果排序（"先运行测试再提交"）和过期
失效，远超沙箱或工具层防护所能表达的范围。

两个工具是互补的。AgentSight 提供运行时可观测：独立的、应用层之下的对 Agent 行为的可见性。ActPlane 提供执行控制平面：确定性的、内核级的关于 Agent 不能
做什么的保证。两者共同实现了三层模型中"验证副作用"层，独立于 harness
提供商，独立于谁拥有沙箱。

两者都是这一架构的可能实现，而非唯一实现。重要的是分离：在环境运营者控制的
层面进行观测和执行控制，无论上面运行的是哪个 Agent 运行时。

这也回应了 Anthropic 在其可信 Agent 框架中指出的生态缺口：对跨部署安全遥测共享和 Agent 安全开放标准的需求。独立的运行时可观测随工作负载
迁移，而不是被锁定在
特定的 harness 或提供商上，这正是两者的基础。

## 实践清单

如果你正在构建或评估 Agent 平台，请在每一层提出这些问题。

**意图授权（MCP / 工具访问）：**

- MCP 服务器是否加入了允许列表？
- OAuth 范围是否最小化且绑定了受众？
- 本地 MCP 服务器是否被视为代码执行风险？
- 高风险工具是否需要人工审批？
- 工具调用是否记录了足够的上下文以供审计？

**执行隔离（沙箱）：**

- 文件系统访问是默认拒绝还是宽泛的工作区挂载？
- Agent 能否访问云元数据端点？
- 网络出口是否按域名、IP 或代理进行限制？
- 服务账号 token 是否挂载到环境中？
- 进程、内存、CPU 和运行时长是否有边界？
- 沙箱策略的拥有者是平台团队还是 Agent 提供商？

**副作用验证（运行时可观测）：**

- 你能否重建 Agent 会话的进程谱系？
- 你能否在框架之下看到文件和凭证访问？
- 你能否将网络出口与 Pod、服务账号和命令关联？
- 你能否检测工具意图与 OS 副作用之间的不匹配？
- 你能否在不仅仅信任框架日志的情况下回放事件？
- 你能否向审计员（SOC 2、ISO 27001）证明自动化 Agent 对生产数据和凭证的访问
  如何被监控和记录？

**护栏集成：**

- 哪些副作用应该被立即阻止？
- 哪些应该触发告警或人工审查？
- 哪些策略属于 MCP 配置、沙箱配置、Kubernetes 策略、eBPF/LSM 还是网络控制？
- 当框架日志和 OS 级可观测不一致时该怎么办？

## 结语

Agent 运行时正变得更强大、更托管化、更不透明。安全模型不能依赖任何单一层，
尤其是当各层有不同的所有者时。

Harness 不是可信的边界。沙箱的所有权取决于部署模型。环境运营者能保证自己
拥有的唯一一层就是 OS/运行时可观测层。

MCP 授权意图。沙箱约束执行。OS 级可观测验证副作用。每一个都是必要的；没有一个
是充分的。实践模型是它们的分离：

```text
授权意图          →  隔离执行            →  验证副作用
（harness 拥有）     （所有权存在争议）     （必须由平台方拥有）
```

实现细节因部署而异，但分离本身以及所有权问题是应该保持稳定的部分。

如果你正在探索这个领域，[AgentSight](https://github.com/eunomia-bpf/agentsight/)
和 [ActPlane](https://github.com/eunomia-bpf/ActPlane) 是我们在观测和执行
控制层分别的开源起点。

## 参考文献

- [GitHub Docs: About Copilot coding agent](https://docs.github.com/en/copilot/concepts/about-copilot-coding-agent)
- [GitHub: Security Architecture of Agentic Workflows](https://github.blog/ai-and-ml/generative-ai/under-the-hood-security-architecture-of-github-agentic-workflows/)
- [GitHub: Copilot coding agent supports self-hosted runners](https://github.blog/changelog/2025-10-28-copilot-coding-agent-now-supports-self-hosted-runners/)
- [OpenAI Codex cloud documentation](https://developers.openai.com/codex/cloud)
- [OpenAI: Run long horizon tasks with Codex](https://developers.openai.com/blog/run-long-horizon-tasks-with-codex)
- [OpenAI: Codex Agent Approvals & Security](https://developers.openai.com/codex/agent-approvals-security)
- [Anthropic 2026 Agentic Coding Trends Report](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf)
- [Anthropic: Trustworthy Agents](https://www.anthropic.com/research/trustworthy-agents)
- [Anthropic Engineering: Claude Code auto mode](https://www.anthropic.com/engineering/claude-code-auto-mode)
- [Anthropic Engineering: Making Claude Code More Secure](https://www.anthropic.com/engineering/claude-code-sandboxing)
- [Anthropic Engineering: Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents)
- [Anthropic NIST RFI on Agentic Security](https://www-cdn.anthropic.com/43ec7e770925deabc3f0bc1dbf0133769fd03812.pdf)
- [Claude Code security documentation](https://docs.anthropic.com/en/docs/claude-code/security)
- [Claude Code permission modes](https://code.claude.com/docs/en/permission-modes)
- [MCP Security Best Practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)
- [MCP Authorization documentation](https://modelcontextprotocol.io/docs/tutorials/security/authorization)
- [Kubernetes SIGs Agent Sandbox](https://agent-sandbox.sigs.k8s.io/)
- [Google Cloud: Agent Sandbox on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/agent-sandbox)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Meta KernelEvolve](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/)
- [SWE-Bench Verified Leaderboard](https://www.vals.ai/benchmarks/swebench)
- [Devin's 2025 Performance Review](https://cognition.ai/blog/devin-annual-performance-review-2025)
- [Goldman Sachs autonomous coder pilot](https://www.cnbc.com/2025/07/11/goldman-sachs-autonomous-coder-pilot-marks-major-ai-milestone.html)
- [Red Hat: Zero trust AI agents on Kubernetes with Kagenti](https://next.redhat.com/2026/03/05/zero-trust-ai-agents-on-kubernetes-what-i-learned-deploying-multi-agent-systems-on-kagenti/)
- [AIDev: Studying AI Coding Agents on GitHub](https://arxiv.org/abs/2602.09185)
- [Agentic Workflow Injection in GitHub Actions](https://arxiv.org/abs/2605.07135)
- [Measuring the Permission Gate: Claude Code Auto Mode](https://arxiv.org/abs/2604.04978)
- [Trustworthy Agentic AI Requires Deterministic Architectural Boundaries](https://arxiv.org/abs/2602.09947)
- [SafeHarness: Security Architecture for LLM-based Agents](https://arxiv.org/abs/2604.13630)
- [SandboxEscapeBench: Can AI Agents Escape Their Sandboxes?](https://arxiv.org/abs/2603.02277)
- [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [Bishop Fox: The Confused Deputy, MCP Attack](https://bishopfox.com/blog/otto-support-confused-deputy)
- [Docker: MCP Horror Stories, GitHub Prompt Injection](https://www.docker.com/blog/mcp-horror-stories-github-prompt-injection/)
- [30+ Vulnerabilities in AI Coding Tools](https://thehackernews.com/2025/12/researchers-uncover-30-flaws-in-ai.html)
- [AI Agent Security Incidents Hit 65% of Firms](https://www.kiteworks.com/cybersecurity-risk-management/ai-agent-security-incidents-2026/)
- [Bessemer: Securing AI agents in 2026](https://www.bvp.com/atlas/securing-ai-agents-the-defining-cybersecurity-challenge-of-2026)
- [InfoQ: Securing Autonomous AI Agents on Kubernetes](https://www.infoq.com/articles/securing-autonomous-ai-agents-kubernetes/)
- [CrowdStrike: What Security Teams Need to Know About OpenClaw](https://www.crowdstrike.com/en-us/blog/what-security-teams-need-to-know-about-openclaw-ai-super-agent/)
- [Reco.ai: The OpenClaw Agent Security Crisis](https://www.reco.ai/blog/openclaw-the-ai-agent-security-crisis-unfolding-right-now)
- [OpenClaw Security Analysis](https://arxiv.org/html/2603.27517v2)
- [SentinelOne: OneClaw Discovery and Observability](https://www.sentinelone.com/blog/oneclaw-discovery-and-observability-for-the-agentic-era/)
- [Epsilla: ClawTrace Agent Observability](https://www.epsilla.com/blogs/clawtrace-launch-openclaw-agent-observability)
- [AgentSight blog post](https://eunomia.dev/zh/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/)
- [AgentSight repository](https://github.com/eunomia-bpf/agentsight/)
- [ActPlane repository](https://github.com/eunomia-bpf/ActPlane)
