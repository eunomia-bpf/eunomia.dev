# 目标读者与用户痛点地图（公开内容版）

> 状态：内部草案，不发布。最后更新：2026-07-19。
> 边界：本文只记录可公开讨论、可长期复用的用户问题和内容判断。不要写客户信息、销售 pipeline、定价、融资、内部交付模型或未验证的竞争情报。

## 为什么单独维护

内容、SEO/GEO 和平台发布的第一层判断应该是用户痛点，而不是平台、关键词或发布格式。这个文档用于回答：谁有问题、他们会怎么寻找答案、现有替代方案哪里不够、我们有什么公开证据可以贡献。

## 核心目标读者

| 读者 | 他们关心什么 | 常见入口 |
|---|---|---|
| AI-agent/runtime builders | agent 实际执行了什么、怎么监控 closed-source CLI、怎么把 intent 和系统行为关联起来 | GitHub、docs、X、HN、Reddit、DEV |
| Platform/SRE teams | agent 接入生产后如何观测、限权、审计、复盘和控制成本 | LinkedIn、Google、docs、case-style blog |
| Security/compliance teams | prompt/tool guardrail 管不到 subprocess、文件、网络和数据流时，怎样得到 runtime evidence 和 enforceable policy | Google、LinkedIn、知乎、HN/Lobsters |
| Kernel/eBPF developers | eBPF 在 AI agent、GPU、runtime extension 里的新用法和边界 | eunomia.dev、GitHub、Lobsters、HN、论文 |
| OSS maintainers/researchers | 可复现 artifact、论文数据、方法、limitations、引用入口 | papers、GitHub、eunomia.dev、arXiv |

## 高价值用户痛点

| 痛点 | 用户会怎么说/搜 | 现有替代方案 | 我们的独特角度 | 对应资产 |
|---|---|---|---|---|
| 看不见 closed-source coding agent 做了什么 | how to monitor Claude Code, AI agent observability without SDKs | OTel/SDK、日志、proxy、手工审计 | 零插桩关联 LLM intent、process、file、network、syscall | AgentSight、agent-session、agentsight.us |
| 不知道 agent 成本和时间花在哪里 | Claude Code cost breakdown, AI agent profiling, semantic flamegraph | 平台账单、LLM trace UI、时间线 | 用 semantic flamegraph 聚合 prompt 意图、token、时间、文件和网络影响 | agentpprof、AgentSight |
| prompt/tool guardrail 覆盖不到真实系统效果 | AI agent runtime security, subprocess bypass guardrails | prompt policy、MCP proxy、approval workflow、sandbox | 把自然语言策略编译成 OS 可观察/可执行规则，覆盖跨事件和信息流 | ActPlane |
| sandbox 之外缺少执行证据 | AI agent sandbox security, agent audit trail | container/microVM/seccomp、人工日志 | 观测和 enforcement 分层：sandbox 管边界，eBPF 记录和约束实际效果 | AgentSight、ActPlane、ACRFence |
| agent 上生产前不知道怎么做治理 | AI agent governance runtime, agent compliance evidence | 商业安全平台、审批流、静态策略 | 提供可审计 trace、policy-as-code、runtime enforcement 和 scoped pilot 入口 | site thesis、ActPlane、AgentSight |
| GPU/系统瓶颈无法从传统视图解释 | GPU observability eBPF, GPU profiling host side stalls | Nsight、CUPTI、driver logs | 把 eBPF/runtime extension 思路扩到 GPU 和 heterogeneous runtime | gpu_ext、gPerf、bpftime GPU |

## 选题优先级判断

优先写能同时满足下面三项的内容：

1. 用户已经在搜索或讨论这个问题。
2. 我们有公开证据，而不是只有观点。
3. 这篇内容能强化一个品牌心智：看得见、管得住、跑得快，或 eBPF/runtime/GPU 系统研究的可信底盘。

降级处理：

- 只有 SEO 词、没有公开证据：先做 research note 或等待数据。
- 只有项目更新、没有用户问题：改成 release note、X/LinkedIn 短帖或暂缓。
- 只有平台热度、没有独特角度：不追。
- 只有 canonical/格式问题：归入发布卫生项，不提升为内容优先级。

## 内容形态映射

| 用户意图 | 最合适内容 | 平台 |
|---|---|---|
| 想解决具体问题 | docs quickstart、FAQ、debugging guide | eunomia.dev docs、DEV、掘金、issue 回链 |
| 想理解机制和边界 | research explainer、architecture article | eunomia.dev、知乎、Medium |
| 想比较方案 | comparison / decision guide | eunomia.dev、LinkedIn、HN/Lobsters |
| 想试工具 | GitHub README、demo page、tutorial | GitHub、docs、Product Hunt 仅限产品时刻 |
| 想参与讨论 | short post、thread、comment | X、LinkedIn、Reddit、HN、Lobsters |

## 发布前问题清单

- 这篇内容解决谁的什么痛？
- 读者会用什么词或在什么社区提出这个问题？
- 他们现在的替代方案是什么，缺口在哪里？
- 我们有哪些公开证据能支撑判断？
- 这篇内容强化哪个品牌心智？
- 长文是否已有中英文 canonical？短帖是否能独立提供一个 insight？
- 发布后的评论、issue 或私信应该引导到哪里？
