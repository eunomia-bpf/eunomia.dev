---
date: 2026-07-15
slug: ebpf-ai-agent-policy-enforcement
description: eBPF 可以阻止 AI agent 的系统影响，但是否允许仍需上下文策略决定。ActPlane 把任务意图、事件历史和 authority 连接到内核执行控制。
---

# 用 eBPF 执行 AI Agent 策略，为什么还需要上下文 Policy Plane

ARMO 最近提出了一个很有价值的问题：面对 AI agent workload，内核级
eBPF enforcement 能抓住什么，又会遗漏什么？它的核心判断是正确的。
eBPF 可以观察和介入进程、文件与网络影响，但一条内核事件无法解释某个
动作背后的任务原因。

实践结论不只是“增加更多上下文”。AI agent security 需要一个 policy
plane，把项目意图、事件历史和 authority 转化为内核可以执行的决策。
内核在已覆盖的影响上提供完整中介，policy plane 提供含义。

[ActPlane 论文](https://arxiv.org/abs/2606.25189)把这种分工变成了具体
设计。论文结果解释了为什么静态 allowlist 和 behavioral baseline 都很
有用，却无法独立表达“只有在最近一次源码修改之后正确测试已经通过，才
允许提交”这样的规则。

<!-- more -->

## 观察、策略和执行控制是三项不同工作

讨论 AI agent runtime security 时，人们经常把三项工作混在一起：

| 工作 | 要回答的问题 | 系统角色 |
|---|---|---|
| 观察 | Agent 及其子进程实际上做了什么？ | 把 agent session 与进程、文件、网络和资源证据关联起来 |
| 策略 | 结合任务、历史和规则 authority，这个影响现在是否允许？ | 解析上下文并维护决策所需状态 |
| 执行控制 | 能否在操作生效前阻止它？ | 用 eBPF 和 BPF-LSM 介入已覆盖的 OS 操作 |

这种分工解释了为什么增加内核 telemetry 无法独立弥合 semantic gap，也
解释了为什么单靠应用层上下文无法覆盖 shell、生成脚本或编译 helper 最终
产生的影响。

完整设计需要连接这些层，同时准确区分各自的保证。
[AgentSight](https://eunomia.dev/zh/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/)
提供系统级 profiling 与 monitoring；ActPlane 接收具体且带有 authority
scope 的规则，再沿真实进程树执行。

## 通用 eBPF Enforcement 会遗漏什么

ARMO 的文章 [eBPF for AI Agent Enforcement: What Kernel-Level Security
Catches and What It
Misses](https://www.armosec.io/blog/ebpf-based-ai-agent-enforcement/)指出了把
传统 runtime security 直接用于 agent 时的两个重要问题。

第一，系统事件只携带很少的意图。eBPF program 可以看见 agent 正在连接
一个陌生 endpoint，但事件不会说明 agent 是在部署到用户批准的服务，还是
在 indirect prompt injection 之后外传凭据。

第二，agent 行为会随任务变化。Server 通常有稳定的进程树和有限的网络
peer。Coding agent 会创建脚本、发现工具，并在不同任务中接触不同文件与
endpoint。静态 allowlist 容易变得过宽而失去保护作用，或者过窄而破坏
agent 的有效自主性。

Behavioral baseline 可以区分常见活动和异常活动，因此有助于 detection。
Authorization 回答的是另一件事：当前任务和策略是否允许这次操作？一次
常见的 `git commit` 仍可能违反仓库规则，因为 agent 在最后一次测试之后
又修改了源文件。一个陌生的部署 endpoint 也可能完全合法，因为用户明确
选择了它。出现频率和 anomaly score 都无法独立给出这两个答案。

## ActPlane 研究揭示了缺失的 Policy Input

ActPlane 研究了开发者已经写入 `CLAUDE.md` 和 `AGENTS.md` 的指令。论文
覆盖 64 个热门仓库、84 份 instruction file 和 2,116 条 statement，并
报告了以下结果：

- 64.3% 的 statement 是行为 directive。
- 83% 的行为 directive 涉及系统可观测行为。
- 81% 的仓库至少包含一条跨事件 directive。
- 74% 的系统可观测 directive 需要项目或任务上下文，才能变成具体规则。

这些结果指出了通用 per-event rule 通常缺少的四种 policy input：

| Policy input | 示例问题 | 重要性 |
|---|---|---|
| 任务上下文 | 当前仓库里的“完整测试”具体是哪条命令？ | 自然语言指令需要解析为具体命令、路径和 endpoint |
| 事件历史 | 最近一次相关写入之后，测试是否成功退出？ | 很多规则描述跨事件的新鲜度、顺序、lineage 或 information flow |
| Authority | 规则由管理员、仓库 owner 还是 task agent 定义？ | 被攻陷的 task agent 不能削弱继承的约束 |
| 恢复反馈 | 操作被拒绝后，agent 必须修复什么状态？ | 语义原因能帮助 agent 遵从策略，减少换路径盲目重试 |

ActPlane 用紧凑 DSL 表达这些 input，沿进程树维护 label 和时序状态，再把
执行状态编译到 eBPF。Higher-authority policy domain 在 task agent 启动
之前加载；子 domain 可以增加或收紧约束，继承规则继续生效。

这是对 ARMO behavioral-baseline 框架的重要补充。Baseline 估计什么是
常见行为，contextual policy 说明什么行为被允许、由谁授权、在任务的哪个
阶段允许。生产系统可以同时使用两者，因为它们回答不同问题。

DSL、label propagation、temporal gate 和部署架构已经在旧文中完整展开，
这里不再重复：[ActPlane：把 Agent Harness Enforcement 下沉到内核
eBPF](https://eunomia.dev/zh/blog/2026/05/31/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf/)。

## 四种 Control Model 如何分工

没有一种 control 能覆盖 agent system 的所有层。更有用的比较方式是看每种
control 能可靠做出哪类决策。

| Control | 任务含义 | 跨事件状态 | 间接 OS 影响 | 主要用途 |
|---|---:|---:|---:|---|
| Prompt 或 tool-call guard | 对拟执行动作较强 | 有限 | Shell-out 后有限 | 执行前引导与工具筛选 |
| 静态 OS allowlist | 除非人工编码，否则没有 | 有限 | 对已覆盖 hook 较强 | 稳定资源边界 |
| Behavioral baseline | 从历史行为推断 | 统计性 | 输入系统 telemetry 时较强 | Detection 与调查 |
| Contextual OS policy | 显式任务和项目上下文 | 显式 | 对已覆盖 hook 较强 | 确定性执行已加载规则 |

这张表不会让 Falco、Tetragon、seccomp、sandbox 或 anomaly detection 失去
价值。它们在各自的决策边界内仍然重要。真正的架构错误，是要求内核事件流
自行推断任务意图，或者要求 tool-call guard 介入它根本看不见的影响。

## ActPlane 结果能支持什么结论

论文评估了从自然语言 directive 到 policy translation、运行时介入、反馈
和 agent 恢复的完整路径。

- 在 190 条 direct、script、hidden 和 compliant trace 上，ActPlane 的 Decision Compliance Rate 为 75.8%，比被评估的 prompt-filter、tool-regex、tool-level IFC 和无反馈 kernel IFC 高 22 到 31 个百分点。
- 使用相同 kernel rule 时，semantic feedback 得到 86 条正确 violation-trace outcome，无反馈时为 27 条。
- 一轮 policy revision 把 violation detection 从 77.2% 提高到 94.7%，说明 policy translation quality 仍然是一等依赖。
- 32 条 active rule 的 no-hit overhead 在 agent-trace replay 上为 1.9%，在 Linux kernel build 上为 6.5%；100 条 rule 时两种 workload 都低于 8.4%。
- 在 361 个 OpenAgentSafety task 中，ActPlane 阻止了 106 个 baseline-unsafe effect 中的 78 个；policy 也在 16% 的 baseline-safe task 上触发，暴露了过宽规则的成本。

这些结果支持一个有边界的结论。对于系统可观测的 OS 影响，包括通过间接
进程路径到达的影响，eBPF 是强 enforcement substrate。最终决策是否正确，
仍然取决于已加载 policy 的质量和 authority。

## 一套可审查的 Runtime Security 架构

AgentSight 与 ActPlane 可以组成下面这条可审查的控制链：

```text
agent 与任务上下文
        ↓
AgentSight 运行时证据与审计
        ↓
operator 或可信 policy agent 审查
        ↓
具体且带有 authority scope 的 policy
        ↓
ActPlane eBPF enforcement
        ↓
返回 agent 的 semantic feedback
```

AgentSight 为 profiling、detection、调查和候选规则审查提供证据。它不会
自动授权或阻止动作。ActPlane 沿已覆盖的进程、文件和网络事件执行已加载
规则。它不会识别每一种恶意 prompt，也不理解任意生成内容。

因此，系统级 runtime safety 仍然是更大架构中的一层。Isolation 限制
blast radius，identity 和 authorization 约束可用能力，内容与协议检查覆盖
syscall event 之外的语义，可信 exception path 处理高影响变更。更完整的
三层模型见旧文：[基于 eBPF 的不透明 AI Agent 运行时可观测与执行控制](https://eunomia.dev/zh/blog/2026/05/25/runtime-security-for-ai-agents/)。

## 常见问题

### eBPF 足以解决 AI Agent Security 吗？

eBPF 对已覆盖 OS event 提供强观察与介入能力。任务意图、policy authority、
内容语义、identity 和 isolation 仍需由内核执行层周围的输入与控制提供。

### Behavioral Baseline 能替代 Policy 吗？

Behavioral baseline 检测偏离历史行为的活动，policy 定义当前任务中的权限。
成熟系统可以用 anomaly detection 提议或排序候选规则，再经过
authority-aware review 后进入 enforcement。

### AgentSight 会执行 ActPlane Policy 吗？

不会。AgentSight 是系统级 profiler 和 monitor，ActPlane 是 policy
enforcement component。运行时证据可以帮助 policy review，但两个项目明确
区分观察与执行控制的职责。

### ActPlane 仍然覆盖不了什么？

纯聊天语义伤害、不安全生成内容、已覆盖 OS hook 之外的 service-side effect、
缺失 hook、kernel compromise 和错误生成的 policy 都需要额外控制。ActPlane
的保证适用于 enforcement engine 所介入的 OS event 上已经加载的规则。
