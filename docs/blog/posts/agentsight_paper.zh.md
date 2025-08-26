---
date: 2025-08-26
---
# AgentSight: 让 AI Agent 的一举一动都在掌控之中，基于 eBPF 的系统级可观测性方案

想象一下，当你的 AI Agent 正在自主编写代码、执行命令时，你却不知道它究竟在做什么，或许有点令人不安？随着 Claude Code、Gemini-cli 等 LLM 智能体工具的普及，我们正在将越来越多的控制权交给 AI。但这里有个棘手的问题：这些 AI 智能体与传统软件截然不同，现有的监控工具要么只能看到它们的"想法"（LLM 提示词），要么只能看到它们的"行动"（系统调用），却无法将两者联系起来。就像你只能听到一个人在说什么，或者只能看到他在做什么，却不知道言行之间的联系。这个盲区让我们难以判断 AI 是在正常工作、遭受攻击还是陷入了昂贵的死循环。

这就是我们开发 AgentSight 的初衷，我们用 eBPF 在系统边界处巧妙地监控 AI Agent 的一举一动。AgentSight 能够拦截加密的 LLM 通信来理解 AI 的意图，同时监控内核事件来追踪它的实际行为，然后用智能引擎将这两条线索关联起来，形成完整的因果链。最棒的是，这一切都不需要修改你的代码，不依赖特定框架，性能开销还不到 3%！在实际测试中，AgentSight 成功检测出了提示注入攻击，及时发现了烧钱的推理死循环，还帮助我们找到了多智能体协作中的性能瓶颈。AgentSight 已经开源，欢迎来 <https://github.com/agent-sight/agentsight> 体验！论文的 arxiv 版本在 <https://arxiv.org/abs/2508.02736> 。
<!-- more -->

## 引言

我们正在见证一场革命：AI 不再只是辅助工具，而是真正的系统参与者。想想看，当你使用 [Claude Code](https://www.anthropic.com/news/claude-code)、[Cursor Agent](https://cursor.com/) 或 [Gemini-cli](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/) 时，你实际上是在让 AI 直接操作你的系统，创建进程、修改文件、执行命令。这些 AI Agent 可以独立完成复杂的软件开发和系统维护任务，这很酷，但也带来了一个令人头疼的问题：我们正在部署的是非确定性的 ML 系统，这给系统的可靠性、安全性和可验证性带来了前所未有的挑战。

最大的问题在于一个关键的语义鸿沟：AI Agent 想做什么（*意图*）和它实际做了什么（*系统动作*）之间存在断层。传统程序的执行路径是可预测的，但 AI Agent 会动态生成代码、创建各种子进程，让现有的监控工具完全摸不着头脑。举个让人细思极恐的例子：假设有个 AI Agent 正在帮你重构代码，它在搜索 API 文档时不小心从外部网站读取到了恶意提示，结果悄悄在你的代码里植入了后门（这就是[间接提示注入攻击](https://arxiv.org/abs/2403.02691)）。应用层监控只能看到一个"成功执行脚本"的记录，系统监控只能看到一个 bash 进程在写文件，谁都意识不到原本的良性操作已经变成了恶意行为，监控工具都成了"睁眼瞎"。

现有的解决方案都卡在了语义鸿沟的某一边。像 [LangChain](https://github.com/langchain-ai/langchain) 和 [AutoGen](https://github.com/microsoft/autogen) 这类框架采用*应用层插桩*，能捕获 AI 的推理过程和工具调用。虽然它们能看到 AI 的*意图*，但这种方法太脆弱了：需要不断追着 API 更新跑，而且轻易就能被绕过，一个简单的 shell 命令就能让它们失明。另一边，*通用系统级监控*倒是能看到所有*动作*，每个系统调用、每次文件访问都逃不过它的眼睛，但问题是它完全不懂上下文。在它看来，一个正在写数据分析脚本的 AI 和一个正在植入恶意代码的 AI 没有任何区别。不了解背后的 LLM 指令，不知道"为什么"，只看到"做了什么"，这些底层事件流就像天书一样难懂。

我们提出了一种全新的监控方法，专门用来弥合这个语义鸿沟。核心洞察很简单：虽然 AI Agent 的内部实现和框架千变万化，但它们与外界交互的接口（内核系统调用、网络通信）是稳定且无法绕过的。通过在这些系统边界上进行监控，我们就能同时捕获 AI 的高层意图和底层系统行为。**AgentSight** 正是基于这个理念，用 eBPF 技术在系统边界处监控：拦截加密的 LLM 通信来获取意图，监控内核事件来追踪实际行为。最精妙的是我们的两阶段关联机制：实时引擎先把 LLM 响应和它触发的系统行为关联起来，然后让一个"观察者" LLM 来分析这些关联数据，推断潜在风险并解释为什么某个行为序列是可疑的。这种方案不需要改你的代码，不依赖特定框架，性能开销还不到 3%，已经成功检测出提示注入攻击、推理死循环和多智能体协作瓶颈。

## 背景与相关工作

在深入技术细节之前，让我们先了解一下 LLM 智能体的工作原理，看看现有的监控方案为什么不够用，以及 eBPF 技术如何成为我们的秘密武器。

现在的 AI Agent 系统基本都遵循一个通用架构，主要包含三大件：（1）负责"思考"的 LLM 后端，（2）负责"执行"的工具框架，（3）负责协调的控制循环。无论是 [LangChain](https://github.com/langchain-ai/langchain)、[AutoGen](https://github.com/microsoft/autogen)、[Cursor Agent](https://cursor.com/)、[Gemini-cli](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/) 还是 [Claude Code](https://www.anthropic.com/news/claude-code)，都是这个模式的变体。这种架构赋予了 AI Agent 强大的能力：你只需要用自然语言描述目标，它就能自主制定计划并执行，比如自己写代码分析数据、调试程序，甚至重构整个项目。

目前的监控方案各有各的盲区。一类是专注于"意图"的工具，像 Langfuse、LangSmith 和 Datadog 这些，它们能很好地追踪 AI 的推理过程和应用层事件，甚至有 OpenTelemetry GenAI 工作组在推动标准化。但问题是，一旦 AI 执行了外部命令或创建了子进程，这些工具就两眼一抹黑了。另一类是专注于"动作"的系统监控工具，比如 Falco 和 Tracee，它们能看到每一个系统调用，但完全不懂这些动作背后的含义，分不清 AI 是在正常工作还是在搞破坏。还有一些研究尝试让 AI 的思维过程更透明，但它们主要关注 LLM 本身的可解释性，并没有解决 AI 意图与系统行为之间的断层问题。

要同时监控网络通信和内核活动，我们需要一个既安全又高效的技术，eBPF（扩展伯克利包过滤器）正好满足所有要求。虽然 eBPF 最初只是用来过滤网络包的，但现在它已经进化成了一个强大的内核虚拟机，支撑着许多现代监控和安全工具。对于 AI Agent 监控来说，eBPF 简直是完美选择：它能在系统边界处精确观察，既能拦截 TLS 加密的 LLM 通信获取*意图*，又能监控系统调用追踪*动作*，而且性能开销极低。更重要的是，eBPF 有内核级的安全保障，包括验证程序终止性和内存安全，这让它可以安心用在生产环境。

## 设计

AgentSight 的核心设计目标很明确：让 AI Agent 的意图和动作能够对应起来。我们通过在系统边界进行监控，配合多信号关联引擎来实现这个目标。

### 守住关键路口

我们发现了一个关键规律：无论 AI Agent 内部怎么变化，它与外界交互的"关口"是固定的，主要就是两个地方：与 LLM 通信的网络边界，以及执行系统操作的内核边界（如图 1）。守住这两个关口，我们就能捕获完整的行为链。这种方法的优势在于：首先是*全覆盖*，从进程创建到文件读写，任何系统操作都逃不过内核的眼睛，即使 AI 创建了子进程也一样；其次是*稳定可靠*，系统调用接口和网络协议的变化速度远比 AI 框架慢得多，这让我们的方案更持久。最重要的是，这种方式不依赖 AI Agent 的"自觉"，而是在系统边界上强制监控，更加安全可靠。

![智能体框架概览](imgs/agent.png)
*图 1：智能体框架概览*

AgentSight 的架构设计很巧妙，同时在两个关键边界布防。如图 2 所示，我们用 eBPF 技术部署了无侵入的探针：一边从 SSL 函数截获解密后的 LLM 通信（包括提示词和响应），了解 AI 的意图；另一边从内核监控系统调用和进程事件，追踪实际行为。然后，我们的关联引擎会把这两条线索串联起来，形成完整的因果链。

![AgentSight 系统架构](imgs/arch.png)
*图 2：AgentSight 系统架构*

让 AgentSight 如此强大的几个关键设计：

**eBPF 提供安全高效的探测能力：** 我们选择 eBPF 不是偶然的，它兼具生产级安全性、极高性能，还能同时访问用户空间和内核数据。通过直接拦截 AI Agent 与 LLM 的解密通信，比传统的网络抓包或代理方案更高效、更简洁。

**多信号关联引擎串联因果链：** 这是整个系统的大脑，负责把意图和动作关联起来。我们采用了三重关联机制：（1）*进程血缘追踪*，通过监控 `fork` 和 `execve` 构建完整的进程树，让子进程的行为也能追溯到源头；（2）*时间关联*，把 LLM 响应后短时间内发生的动作关联起来；（3）*内容匹配*，直接匹配 LLM 响应中提到的文件名、URL、命令等与实际系统调用参数。三管齐下，让 AI 的意图和行为之间的因果关系无处遁形。

**"AI 监督 AI"的语义分析：** 传统的基于规则的检测太死板，我们让另一个 LLM 来当"安全分析师"。把关联好的事件追踪喂给它，利用 LLM 理解复杂语义、推断因果关系的能力，发现那些不符合预定义模式的异常行为。这种"以 AI 制 AI"的方法，让 AgentSight 的检测能力更加智能和灵活。

## 实现

AgentSight 的实现包括一个用 Rust/C 编写的用户空间守护进程（6000 行代码）来协调 eBPF 程序，加上一个 TypeScript 前端（3000 行）用于可视化分析。整个系统为高性能而生，能够实时处理海量的内核事件并转换成人类可读的追踪数据。

### 在边界处布下天罗地网

我们的 eBPF 探针就像两个哨兵，分别守在不同的关口。第一个哨兵通过 uprobes 挂载到 OpenSSL 等加密库的 SSL_read/SSL_write 函数上，截获解密后的 LLM 通信内容。为了处理像 SSE 这样的流式协议，我们还实现了智能的数据重组机制。第二个哨兵则守在内核层，通过 sched_process_exec 等追踪点构建进程树，用 kprobes 监控 openat2、connect、execve 等关键系统调用。为了避免数据洪流，我们在内核层就进行精准过滤，只把目标 AI Agent 的事件送到用户空间，大大减少了性能开销。

### 两阶段关联引擎：从数据到智能

我们用 Rust 实现的关联引擎分两步走。第一步是实时处理：从 eBPF 环形缓冲区读取事件，补充上下文信息（比如把文件描述符转换成实际路径），维护进程树状态，并在 100-500 毫秒的时间窗口内进行因果关联。第二步是语义分析：把关联好的事件整理成结构化日志，构造精心设计的提示词，让辅助 LLM 扮演"安全分析师"角色进行深度分析。LLM 会输出自然语言的分析结果和风险评分。为了控制延迟和成本，我们采用异步处理和优化的提示工程技术。

## 评估

我们主要想验证两件事：AgentSight 会不会拖慢系统？它能不能真正发现问题？

我们在真实环境中测试了 AgentSight 的性能影响。测试环境是 Ubuntu 22.04（Linux 6.14.0）服务器，用 Claude Code 1.0.62 作为测试对象。我们选了三个典型的开发场景，都基于[这个教程仓库](https://github.com/eunomia-bpf/bpf-developer-tutorial)：让 AI 理解整个代码库（`/init` 命令）、生成 bpftrace 脚本、并行编译整个项目。每个场景都测试了 3 次，对比开启和关闭 AgentSight 的运行时间。

| 任务 | 基准线 (秒) | AgentSight (秒) | 开销 |
|------|--------------|----------------|----------|
| 理解仓库 | 127.98 | 132.33 | 3.4% |
| 代码编写 | 22.54 | 23.64 | 4.9% |
| 仓库编译 | 92.40 | 92.72 | 0.4% |

*表 1：AgentSight 引入的开销*

结果很令人满意！如表 1 所示，AgentSight 的平均性能开销只有 2.9%，几乎感觉不到。最重的代码编写场景也只慢了不到 5%，而编译这种 I/O 密集型任务几乎没影响。

我们通过三个真实案例来展示 AgentSight 的威力。

#### 案例 1：抓住提示注入攻击

我们模拟了一个[间接提示注入攻击](https://arxiv.org/abs/2403.02691)场景。一个数据分析 AI Agent 收到了看似正常的任务请求，但其中暗藏恶意指令，最终诱导它泄露了系统的 `/etc/passwd` 文件。

AgentSight 完美捕获了整个攻击过程：从 AI 与恶意网页的初始交互，到创建子进程、建立可疑连接，再到最终读取敏感文件，整条攻击链清清楚楚。我们的"安全分析师" LLM 看到这些关联数据后，立即给出了最高风险评分（5/5）。它的分析一针见血：一个声称要"分析销售数据"的 AI，却在执行 shell 命令读取系统密码文件，还连接到陌生域名，这明显是提示注入攻击的典型特征。这个案例完美展示了意图和动作关联的价值。

#### 案例 2：及时止损的推理死循环

一个 AI Agent 在处理复杂任务时犯了个常见错误：用错参数调用命令行工具。更糟的是，它没能从错误中学习，而是不断重复同样的错误命令。

AgentSight 的实时监控立即发现了异常：短时间内 12 次 API 调用，都在重复同样的模式。"安全分析师" LLM 一眼看出了问题所在：AI 陷入了"尝试-失败-再尝试"的死循环，每次都执行同样的错误命令，收到同样的报错，却始终没能理解错误信息。在循环了 3 轮（可配置的阈值）后，系统自动触发了警报。这时 AI 已经烧掉了 4,800 个 token，相当于 2.40 美元。及时的干预避免了更大的损失，这就是智能监控的价值所在。

#### 案例 3：揭秘多 Agent 协作瓶颈

我们监控了一个由三个 AI Agent 组成的开发团队，它们协作完成软件开发任务。AgentSight 记录了惊人的 12,847 个事件，从中发现了有趣的协作模式：Agent B 有 34% 的时间在等待 Agent A 完成设计，而 A 的频繁修改又导致 B 不断返工。通过分析，我们发现虽然这些 AI 自发形成了一些协调机制，但如果采用更明确的协作策略，可以减少总运行时间。这个案例展示了系统边界监控的独特价值：传统的应用层监控根本看不到这种跨进程的协作动态。

## 结论

AI Agent 正在改变软件开发和系统运维的方式，但它们的不可预测性和复杂性也带来了前所未有的挑战。AgentSight 通过在系统边界进行创新的监控，成功解决了 AI 意图与系统行为之间的断层问题。

我们用 eBPF 在系统边界上布下天罗地网，既能截获加密的 LLM 通信了解 AI 在"想什么"，又能监控内核事件追踪 AI 在"做什么"，然后用智能的关联引擎把两者串联起来。实践证明，AgentSight 不仅能有效检测提示注入攻击、发现推理死循环、优化多 Agent 协作，而且性能开销不到 3%，真正做到了"看得清、管得住、不拖累"。

这种"以 AI 制 AI"的监控方案，为 AI Agent 的安全部署提供了坚实保障。随着 AI Agent 变得越来越强大和自主，像 AgentSight 这样的可观测性工具将变得越来越重要。我们相信，只有真正理解和掌控 AI 的行为，才能放心地让它们承担更多责任。

**AgentSight Github：** <https://github.com/agent-sight/agentsight>

---

## 参考文献

1. **claudecode**: Anthropic. "Introducing Claude Code." Anthropic Blog, Feb 2025. https://www.anthropic.com/news/claude-code

2. **cursor**: Anysphere Inc. "Cursor: The AI‑powered Code Editor." 2025. https://cursor.com/

3. **geminicli**: Mullen, T., Salva, R.J. "Gemini CLI: Your Open‑Source AI Agent." Google Developers Blog, Jun 2025. https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/

4. **indirect-prompt-inject**: Zhan, Q., Liang, Z., Ying, Z., Kang, D. "InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents." ACL Findings, 2024. https://arxiv.org/abs/2403.02691

5. **langchain**: Chase, H. "LangChain: Building applications with LLMs through composability." 2023. https://github.com/langchain-ai/langchain

6. **autogen**: Wu, Q., et al. "AutoGen: Enable Next-Gen Large Language Model Applications." Microsoft Research, 2023. https://github.com/microsoft/autogen

7. **Maierhofer2025Langfuse**: Maierhöfer, J. "AI Agent Observability with Langfuse." Langfuse Blog, March 16, 2025. https://langfuse.com/blog/2024-07-ai-agent-observability-with-langfuse

8. **langfuse**: "Langfuse - LLM Observability & Application Tracing." 2024. https://langfuse.com/

9. **langsmith**: LangChain. "Observability Quick Start - LangSmith." 2023. https://docs.smith.langchain.com/observability

10. **Datadog2023Agents**: Datadog Inc. "Monitor, troubleshoot, and improve AI agents with Datadog." Datadog Blog, 2023. https://www.datadoghq.com/blog/monitor-ai-agents/

11. **helicone**: "Helicone / LLM-Observability for Developers." 2023. https://www.helicone.ai/

12. **Liu2025OTel**: Liu, G., Solomon, S. "AI Agent Observability -- Evolving Standards and Best Practices." OpenTelemetry Blog, March 6, 2025. https://opentelemetry.io/blog/2025/ai-agent-observability/

13. **Bandurchin2025Uptrace**: Bandurchin, A. "AI Agent Observability Explained: Key Concepts and Standards." Uptrace Blog, April 16, 2025. https://uptrace.dev/blog/ai-agent-observability

14. **Dong2024AgentOps**: Dong, L., Lu, Q., Zhu, L. "AgentOps: Enabling Observability of LLM Agents." arXiv preprint arXiv:2411.05285, 2024.

15. **Moshkovich2025Pipeline**: Moshkovich, D., Zeltyn, S. "Taming Uncertainty via Automation: Observing, Analyzing, and Optimizing Agentic AI Systems." arXiv preprint arXiv:2507.11277, 2025.

16. **falco**: The Falco Authors. "Falco: Cloud Native Runtime Security." 2023. https://falco.org/

17. **tracee**: Aqua Security. "Tracee: Runtime Security and Forensics using eBPF." 2023. https://github.com/aquasecurity/tracee

18. **Rombaut2025Watson**: Rombaut, B., et al. "Watson: A Cognitive Observability Framework for the Reasoning of LLM-Powered Agents." arXiv preprint arXiv:2411.03455, 2025.

19. **Kim2025AgenticInterp**: Kim, B., et al. "Because we have LLMs, we Can and Should Pursue Agentic Interpretability." arXiv preprint arXiv:2506.12152, 2025.

20. **brendangregg**: Gregg, B. "BPF Performance Tools." Addison-Wesley Professional, 2019.

21. **ebpfio**: eBPF Community. "eBPF Documentation." 2023. https://ebpf.io/

22. **cilium**: Cilium Project. "eBPF-based Networking, Observability, and Security." 2023. https://cilium.io/

23. **kerneldoc**: Linux Kernel Community. "BPF Documentation - The Linux Kernel." 2023. https://www.kernel.org/doc/html/latest/bpf/

24. **ebpftutorial**: eunomia-bpf. "eBPF Developer Tutorial." GitHub, 2024. https://github.com/eunomia-bpf/bpf-developer-tutorial
