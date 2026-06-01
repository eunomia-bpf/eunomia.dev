---
date: 2026-02-17
description: AgentCgroup 系统性刻画 AI 编程 Agent 的操作系统资源行为，分析 144 个 SWE-rebench 任务，并说明为什么需要基于 eBPF 的内核态细粒度资源控制。
---

# AgentCgroup：当 AI 编程 Agent 遇到操作系统资源

想象一个 AI 编程 Agent 被丢进云上的沙箱容器：它先读代码，再改文件，接着运行测试；测试失败后继续修改、继续跑；中途还可能安装依赖、调用脚本、拉起子进程。对用户来说，这只是一次“帮我修 bug”的请求；对操作系统来说，它是一段持续数分钟、不断改变资源需求、并且很难提前预测的工作负载。

Claude Code、OpenHands、SWE-agent 这类 AI 编程 Agent 正越来越多地被部署到多租户云环境中，在沙箱容器内执行多样化的工具调用。问题是，尽管这类工作负载采用得越来越广，它们在操作系统层面的资源动态仍然缺少系统性理解。我们做了第一次系统刻画：基于 SWE-rebench benchmark，跨两个 LLM 后端分析 144 个真实软件工程任务。测量结果显示，包含容器初始化和工具执行在内的 OS 层开销占端到端延迟的 56-74%，LLM 推理本身只占 26-44%。内存呈现 15.4x 的峰值/平均值比，而 serverless 通常约为 1.5x，微服务通常为 2-3x；内存变化率在亚秒级 burst 中可达到 3 GB/s。同一种工具类型 Bash 的内存消耗会因为命令语义不同而相差 13.7x；同一个任务重复运行也会出现 1.8x 的执行时间差异，并且 token 输出量与峰值内存几乎没有相关性（r = -0.14）。

真正麻烦的地方在于，这些特征刚好踩在现有资源管理机制的盲区上。从内核 cgroup limits、systemd-oomd 到 Kubernetes VPA，静态资源分配要么浪费 93% 的已配置容量，要么触发 OOM kill，摧毁 Agent 已经积累数分钟、且无法确定性复现的执行状态。本文总结我们的 [AgentCgroup paper](https://github.com/yunwei37/agentcgroup-paper) 中的刻画结果，并说明基于 eBPF 的内核态执行控制如何弥合 Agent 工作负载动态与 OS 资源控制之间的 gap。

<!-- more -->

> Paper: *AgentCgroup: Understanding and Controlling OS Resources of AI Agents*
>
> GitHub: [https://github.com/yunwei37/agentcgroup-paper](https://github.com/yunwei37/agentcgroup-paper)

## 我们做了什么

为了把这个问题从“感觉上很不稳定”变成可以分析的数据，我们对生产级 AI 编程 Agent [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 进行了插桩，运行来自 [SWE-rebench](https://github.com/swe-bench/SWE-ReB) benchmark 的 144 个软件工程任务，并覆盖两个 LLM 后端：

- Claude Haiku 4.5（cloud API）：LLM 推理运行在 Anthropic 云端；容器内只运行 Agent framework 和工具调用。
- GLM-4.7-Flash（local GPU）：LLM 推理运行在本地 GPU；所有执行都发生在同一台机器上。

两组实验使用完全相同的 Agent framework（Claude Code，基于 Node.js）。唯一不同的是底层模型，以及推理发生在云端还是本地。这让我们能够隔离模型选择对容器层资源动态的影响。

### 实验设置

| Component | Details |
|-----------|---------|
| Platform | Intel Core Ultra 9 285K（24 cores, 5.8 GHz），128 GB DDR5，Ubuntu 24.04.3 LTS |
| Kernel | Linux 6.15.11，启用 cgroup v2 |
| Container Runtime | Podman（rootless，隔离容器） |
| Agent Framework | Claude Code（Node.js） |
| Models | Haiku 4.5（cloud API，33 个任务）+ GLM-4.7-Flash（local GPU，111 个任务） |
| Benchmark | SWE-rebench（来自开源项目的真实 GitHub issues） |
| Monitoring | 通过 `podman stats` 以 1 秒间隔采样 CPU/内存 |
| Tracing | 从 Agent 执行 trace 中提取工具调用边界（类型、开始/结束时间戳） |

刻画阶段没有施加任何资源限制，以捕获不受约束的、ground-truth 的真实资源行为。

### 任务覆盖

我们的数据集覆盖 6 类任务、3 个难度等级，代表真实软件工程场景：

| Category | Example Projects | Difficulty Levels |
|----------|------------------|-------------------|
| CLI Tools | faker, click | Easy, Medium, Hard |
| DevOps / Build | pre-commit, dvc | Easy, Medium, Hard |
| ML / Scientific | numba, scikit-learn | Easy, Medium, Hard |
| Medical / Bio | pydicom, biopython | Easy, Medium, Hard |
| SQL / Data | sqlalchemy, pandas | Easy, Medium, Hard |
| Web / Network | streamlink, requests | Easy, Medium, Hard |

此外，我们还整理了一个 18 个任务的代表性子集（6 类任务 x 3 个难度），用于更细粒度的按类别分析。

## 刻画结果

先看完整图景。我们围绕三个维度组织刻画结果；每个维度都关系到资源控制中的一个不同方面：

1. 执行模型：决定资源变化应该以什么粒度管理。
2. 资源动态：决定控制逻辑需要以多快速度响应。
3. 不可预测性：决定资源需求能否提前预测。

### Agent 执行模型

第一件需要纠正的直觉是：Agent 慢，不一定是因为模型在慢。很多时候，真正占时间的是容器、工具和操作系统基础设施。

#### 主导延迟的是 OS 基础设施，不是 LLM 推理

与“LLM 才是瓶颈”这个直觉相反，我们的测量显示，LLM 推理只占端到端任务延迟的 26-44%。剩下的 56-74% 都是 OS 层开销：

| Latency Component | Haiku | GLM |
|-------------------|-------|-----|
| Container + agent initialization | 47.7% | 31.0% |
| Tool execution | 25.9% | 25.5% |
| LLM reasoning | 26.4% | 43.5% |

单是容器启动平均就需要 26.5 秒（中位数 23.0 秒，最大 97 秒）。主要原因是 Podman 对 overlay layers 进行 user-namespace ID remapping，这个过程随镜像大小增长。SWE-rebench 的容器镜像大小在 2.9 GB 到 17.3 GB 之间（中位数 3.5 GB），大约是典型微服务镜像的 7 倍、serverless function 的 70 倍，因此初始化开销非常可观。

因此，优化基础设施，尤其是容器启动和工具执行期间的资源调度，可以直接改善超过一半的用户感知完成时间。只关注 LLM inference 优化会错过端到端延迟中更大的那部分。

#### 任务时长与状态性

每个 Agent 任务运行 5-11 分钟（GLM 平均 10.8 分钟，Haiku 平均 5.8 分钟，整体中位数 8.1 分钟），并在单个容器内执行有状态的多轮推理和工具调用循环。Agent 任务的持续时间介于 serverless invocation（100ms-2s）和 batch job 之间，但它又是进程内有状态的：所有 LLM context、中间代码修改和工具结果都保存在进程内存中。

#### 工具执行构成

Bash 和 sub-agent（Task）调用主导工具执行时间，在两个模型中都占工具时间的 90% 以上。不过，两个模型采取的策略很不一样。

Haiku 把工作分散到多种工具类型：

- Sub-agent calls（Task）：47.8% 的工具时间（平均每次调用 100.47 秒）
- Bash：43.2%（平均每次调用 3.76 秒）
- WebSearch/WebFetch：约 5%
- Read、Edit、Grep：合计 <5%

GLM 几乎把所有工作都集中在 Bash：

- Bash：99.5% 的工具时间（平均每次调用 5.93 秒）
- 不使用 sub-agent 或 web search

这种差异对资源管理有直接影响。Haiku 会把一部分计算卸载给外部服务（sub-agents、web search），而 GLM 会把几乎所有计算都压到本地 Bash 调用上，因此本地资源消耗显著更高（Bash 总时间：GLM 为 19,598 秒，Haiku 为 1,543 秒）。

#### Bash 命令语义

并不是所有 Bash 调用都一样。按命令语义拆分后：

| Bash Category | % of Bash Time (Haiku) | % of Bash Time (GLM) |
|---------------|------------------------|----------------------|
| Test execution（pytest、unittest 等） | 72.9% | 43.7% |
| Python snippets | n/a | 26.9% |
| Package installation | 10.8% | 10.1% |
| Git operations | <5% | <5% |
| File exploration | <5% | <5% |

测试执行占据绝对主导地位。而下一节会看到，它同时也是资源消耗最重的类别。

#### “理解-修改-验证”的时间模式

如果把一次执行过程分成 10 个等长阶段，并绘制工具分布随时间的变化，会出现一个清晰模式：

- Understand phase（0-30%）：Read 操作占主导，用于代码探索。
- Modify phase（30-70%）：Edit 操作分布在整个中段；Bash 开始上升。
- Verify phase（40-100%）：Bash 达到高峰，主要来自反复测试执行和调试。

这个阶段特征与人类软件工程中的“理解、修改、验证”流程一致，也为 phase-aware resource control 提供了依据。

### 资源动态

知道时间花在哪里之后，下一个问题是资源怎么变化。这里最反直觉的结论是：CPU 不是主要瓶颈，内存才是。

#### 并发瓶颈是内存，不是 CPU

Agent 的 CPU 利用率很低：

| Metric | Haiku | GLM |
|--------|-------|-----|
| Average CPU utilization | 13.2% | 7.6% |
| Samples exceeding 50% CPU | 8.2% | 0.5% |
| Peak CPU | >175%（multi-core） | >100%（短暂 spike） |

在我们的 24 核平台上，即使达到内存限制下的最大并发密度，CPU 也保持在 36% 以下。内存则完全不同：每个任务的峰值内存可达到 2-4 GB。按峰值分配时，128 GB RAM 只能支持 32-64 个并发实例，而 CPU 仍然远未充分利用。

这种 CPU-memory imbalance 意味着，动态内存管理是提升多租户密度的关键杠杆：在短暂内存 burst 期间弹性扩展，在 idle 阶段回收资源，从而容纳更多并发实例。

#### “双层”内存结构

Agent 内存呈现一种独特的双层结构；我们在以往任何工作负载刻画中都没有观察到这种模式。

第一层是 framework baseline（约 185 MB）：Node.js runtime、V8 JIT cache 和 Agent framework 状态会在整个执行过程中维持一个稳定且不可压缩的内存底座，即使处于没有工具活动的 LLM reasoning 阶段也是如此。在全部 144 个任务中，执行早期内存平均值为 183 MB（Haiku）和 188 MB（GLM）。

第二层是 tool-call bursts（500 MB 到 2+ GB）：测试执行、依赖安装和数据处理会制造短暂 spike，通常只持续 1-2 秒，然后回落到约 185 MB 的 baseline。

当我们按执行进度归一化并聚合全部 144 个任务的内存 trace 时，模式很清楚：前半段执行稳定在 185-200 MB baseline；后半段方差逐渐增大并出现大 spike，对应 Bash 密集的 verify phase。

在多租户部署中，64 个并发实例仅 framework baseline 就需要约 12 GB 内存。叠加在 baseline 之上的 tool-call bursts 才是真正的资源管理挑战，而且它们需要与稳定 baseline 不同的处理方式。

#### 98.5% 的内存 burst 由工具调用驱动

我们把每个 1 秒资源采样点标注为“during tool call”或“during LLM reasoning”，并统计超过 300 MB（约为 framework baseline 的 1.6x）的内存 burst：

| Metric | Haiku | GLM |
|--------|-------|-----|
| Tool call time fraction | 50.6% | 35.9% |
| Memory bursts during tool calls | 98.5% | 67.3% |
| Burst concentration ratio | 1.9x | 1.9x |
| CPU bursts during tool calls | 55.3% | 30.2% |

这里的不对称性值得注意。内存 burst 几乎完全由工具调用驱动，而 CPU burst 更分散；GLM 的本地 GPU inference 即使在工具调用之外也会产生稳定 CPU 负载。这意味着内存应该按工具调用粒度管理，而 CPU 需要更宽的上下文感知。

#### 亚秒级 burst 与巨大的峰值/平均值比

资源 burst 不仅由工具驱动，而且非常短暂：

- 最大内存变化率：3 GB/second
- 最大 CPU 变化率：>50%/second
- burst 持续时间：通常 1-2 秒

我们观察到的最高案例是一个 pydicom bioinformatics 任务（Medical_Bio_Hard）：峰值内存达到 4060 MB，而平均值只有 264 MB，峰值/平均值比为 15.4x。这个 4 GB spike 大约只持续 1-2 秒，然后回落到 230 MB baseline。

与传统云工作负载对比：

| Workload Type | Typical Peak/Avg Ratio |
|---------------|------------------------|
| Serverless / FaaS | ~1.5x |
| Microservices | 2-3x |
| Batch / HPC | ~1x |
| AI Coding Agent | up to 15.4x |

这个比例让静态资源限制变得不现实。按峰值分配（4060 MB）意味着 98% 的时间内存使用低于 264 MB，造成 93% 浪费。按平均值分配（264 MB）则会在工具 burst 时触发 OOM kill，摧毁所有 Agent 状态。没有一个单一静态阈值可以同时容纳低 baseline 和短暂 spike。

#### 同一种工具，完全不同的资源消耗

一个有意思的发现是，同一种工具类型（Bash）的资源消耗会因为它实际运行的内容不同而相差 13.7x。资源需求由命令语义决定，而不是工具类型决定：

| Bash Category | P95 Memory Spike (Haiku) | P95 Memory Spike (GLM) | Avg CPU Spike |
|---------------|--------------------------|------------------------|---------------|
| Test execution（pytest 等） | 518 MB | 234 MB | +3.2% |
| Package installation | 233 MB | n/a | moderate |
| Git operations | 13.5 MB | n/a | minimal |
| File exploration | 4.5 MB | n/a | minimal |

Medical/bioinformatics Bash 命令平均峰值内存为 4 GB；web/network 命令平均为 291 MB，相差 13.7x。同一个 `Bash` 工具调用可以只是一个简单的 `ls`，也可以是完整的 `pytest` suite 并加载数 GB 测试数据。基于工具类型的资源策略因此无效；必须理解实际执行的命令语义。

#### CPU 与内存相互独立

CPU 和内存并不会一起变化。不同任务中 CPU 使用率与内存使用量的相关性从 -0.84 到 +0.50 不等，均值为 -0.39。一些任务呈正相关（工具执行同时拉高 CPU 和内存），另一些任务呈负相关（CPU 密集阶段恰好对应较低内存）。这种任务相关的耦合关系意味着资源控制不能假设 CPU 和内存需求协同变化，而必须独立监控和管理两个维度。

### 不可预测性

如果资源需求虽然 spiky 但可以提前预测，调度器仍然有机会做静态规划。但 Agent 的难点在于，同一个任务、同一个框架、甚至同一个工具类型，都可能跑出完全不同的资源轨迹。

#### 同一个任务内部也有非确定性

我们把完全相同的任务（iterative/dvc#777）运行三次，得到：

| Run | Execution Time | Solution Strategy |
|-----|----------------|-------------------|
| 1 | 402 seconds | Strategy A（不同文件修改） |
| 2 | 222 seconds | Strategy B（不同方法） |
| 3 | 259 seconds | Strategy C（不同文件数量） |

这意味着执行时间有 1.8x 差异，而且每次的解题策略完全不同。这种非确定性来自 LLM reasoning 的随机性和决策路径多样性：Agent 每次运行都可能选择完全不同的代码修改、工具序列和调试路径。

#### Token 数量不能预测资源使用

我们分析了 LLM 可观测 proxy 与实际资源消耗之间的相关性：

| Proxy to Target | Haiku (r) | GLM (r) |
|-----------------|-----------|---------|
| Output tokens to peak memory | -0.14 | +0.02 |
| Conversation rounds to execution time | +0.57 | +0.82 |
| Conversation rounds to peak memory | +0.02 | +0.11 |

输出 token 数与峰值内存几乎没有相关性。即使 conversation rounds 对执行时间有中等预测能力，它也无法预测内存。资源消耗由实际执行的工具决定，例如 pytest 还是 file read，而不是由 LLM reasoning 的规模决定。这意味着，即使能够预测一个 Agent 会“思考”多久，也仍然无法预测它需要多少内存。

#### Retry loop 与渐进式内存积累

Retry 行为是 Agent 工作负载的典型特征，而传统容器化应用没有对应模式：

| Metric | Haiku | GLM |
|--------|-------|-----|
| Tasks with retry loops（3+ consecutive identical Bash calls） | 85% (28/33) | 97% (108/111) |
| Average retry groups per task | n/a | 3.9 |
| Maximum consecutive retries | n/a | 56 |
| Execution time consumed by retries | 7.4% | 20.5% |

“执行测试、观察失败、修改代码、再次测试”的迭代循环是 Agent 的行为签名。每次 retry 都会保留之前的内存上下文而不做清理，导致渐进式内存积累。在我们观察到的最坏情况下，未释放内存最高达到 502 MB。这意味着，执行早期足够的内存限制，可能会在后续 retry 积累后触发 OOM kill。

#### 跨任务异质性

在我们的数据集中，峰值内存需求从 197 MB 到 4 GB 不等（coefficient of variation = 147%）：

- Scientific computing tasks（numba、pydicom）：2-4 GB
- CLI tools（faker）：约 200 MB
- Network utilities（streamlink）：约 300 MB

即使使用同一个 Agent framework，不同任务之间仍有 20x 差异。模型选择会进一步放大差异：在相同任务上，Haiku 和 GLM 的 CPU 利用率相差 1.7x。只替换底层模型，而不改变 Agent framework，就会产生完全不同的资源 profile。

### Agent 工作负载与传统云工作负载的对比

| Dimension | Serverless | Microservices | Batch/HPC | AI Coding Agent |
|-----------|------------|----------------|-----------|-----------------|
| Duration | 100ms-2s | Long-running | Min-hours | 5-11 minutes |
| Statefulness | Stateless | External state | Stateful | In-process stateful |
| Memory peak/avg | ~1.5x | 2-3x | ~1x | 15.4x |
| CPU pattern | Brief spike | 10-40% steady | 80-100% | <13% avg, >175% peaks |
| Determinism | Deterministic | Mostly | Deterministic | 1.8x variance same task |
| Resource pattern | Flat | Steady + daily cycles | Stable rise | Burst-silence alternating |
| Kill cost | Just retry | Migrate | Lose progress | Lose all LLM context |
| Image size | ~50 MB | ~500 MB | Varies | 3.5 GB median |

简而言之，Agent 工作负载太有状态，不能随便 kill；太 spiky，不能简单 cap；太不可预测，不能依赖预测；又太短，无法摊薄容器开销。

## 三个错位

把这些现象放在一起看，就会发现问题不是“把内存 limit 调大一点”这么简单。Agent 工作负载与现有资源管理栈之间至少有三类错位。

### 1. 粒度错位

容器级策略（cgroup `memory.max`、Kubernetes QoS）会给整个容器设置一个统一阈值，但 Agent 资源需求是在工具调用粒度变化的。一个 `git status`（13.5 MB spike）和一次 `pytest` 运行（518 MB spike）需要完全不同的内存预算，却共享同一个 cgroup。`memory.high` soft limit 无法区分约 185 MB 的 framework memory（不可压缩的 Node.js heap、V8 JIT cache）和工具子进程内存（可压缩、可限制）。当内核 reclaim 命中 framework pages 时，会造成 V8 GC 压力和 JIT cache 抖动，从而降低 LLM response parsing 的性能。

### 2. 响应性错位

用户态 controller（systemd-oomd、Meta oomd、Kubernetes VPA）的响应时间尺度从毫秒到分钟不等。Agent 内存 burst 持续 1-2 秒，变化率可达 3 GB/s。完整的 PSI signal 到用户态 daemon，再到决策，再到 cgroup write-back 的路径，最理想也要几十毫秒。到那时，burst 可能已经结束，或者已经触发 kernel OOM kill。VPA 以 Pod restart 粒度调整（分钟级）；即使 in-place resize（alpha）也是分钟级。它们都无法在单次工具调用内及时响应。

### 3. 适应性错位

基于历史的预测（Google Autopilot、Kubernetes VPA percentile recommendations）假设工作负载可以复现。但 Agent 的非确定性打破了这个假设：同一个任务执行时间相差 1.8x，解题策略完全不同；token-to-memory correlation 为 0（r = -0.14）；跨任务差异达到 20x。过去运行的 P95 并不是未来运行的可靠上界。而且不同于 serverless，kill-restart 只损失 100ms；kill 一个 Agent 会摧毁 5-11 分钟积累出来的有状态上下文，而且这些上下文不能确定性复现。

## AgentCgroup：基于 eBPF 的内核态资源控制

AgentCgroup 的出发点是：不要把 Agent 当作一个普通长跑容器，也不要把每次工具调用都压进同一个静态 limit。为了解决这三类错位，我们提出 AgentCgroup：一个基于 eBPF 的资源 controller，并对应采用三条设计原则。

### 细粒度资源域（解决粒度错位）

AgentCgroup 使用层次化 cgroup v2 结构组织资源。每个 Agent workload 映射为一个 cgroup node，工具调用作为其 child nodes。这样既可以维持整体 workload budget，也可以为每次工具调用设置资源约束。恢复方面，AgentCgroup 使用 cgroup v2 lifecycle primitives：当工具调用超过 soft limit 时冻结子树；确实需要终止时，原子化 kill 子树，而不是 kill 整个 Agent。

### 内核态执行控制（解决响应性错位）

AgentCgroup 通过 eBPF 直接在内核 cgroup enforcement points 执行控制逻辑，实现微秒级响应，避免用户态/内核态 round trip：

- CPU 方面，`sched_ext` 在 BPF maps 中维护每个 workload 和每个工具调用的 metadata，优先调度 latency-sensitive tool calls，并在错误时自动 fail-safe reversion。
- 内存方面，`memcg_bpf_ops` hooks 在 cgroup 突破 soft limit（`memory.high`）时实现自定义 throttling delays，并以 `memory.max` 作为 hard limit。

### 运行时自适应策略（解决适应性错位）

AgentCgroup 不依赖历史预测，而是使用 eBPF 在内核中追踪进程创建和内存分配，实时检测工具调用边界和资源动态。当内存压力上升时，BPF 程序会采用渐进式响应，例如通过 `memory.high` delay 进行 throttling，或通过 `cgroup.freeze` 冻结，而不是直接终止，从而保留 Agent 状态。

### 初步结果

我们在 patched Linux 6.19.0-rc5 kernel（bpf-next + memcg struct_ops RFC patches）上，以 50x 加速重放真实 Agent memory traces，在多租户设置中评估 AgentCgroup。三个并发 Agent trace 共享受限内存：

Tight memory scenario（总内存 1100 MB，combined demand 约 1233 MB）：

- Baseline：OOM-kill 一个低优先级进程（66% survival）
- AgentCgroup：所有进程完成（100% survival），触发 239 次 throttle，高优先级 Agent 只增加 +2.8% overhead

Moderate memory scenario（总内存 1300 MB）：

- AgentCgroup 通过减少内存争用，将高优先级 P95 allocation latency 降低 29%（70.97 ms 到 50.14 ms）
- P50 latency overhead：+0.3%
- Total completion time：-1.1%（净改善）

执行控制开销可以忽略不计，BPF throttling precision 的相对误差在 2.3% 以内。

## 展望

这只是第一步。当前评估基于 trace replay 和 proof-of-concept prototype；刻画覆盖了一个 Agent framework（Claude Code）和一个 benchmark（SWE-rebench）。还有很多方向值得继续探索：

- 在生产规模下使用真实并发工作负载进行 live agent evaluation
- 覆盖更多 Agent frameworks（OpenHands、SWE-agent、Cursor）和 coding 之外的领域
- 在不同 container runtimes（Docker、gVisor、microVMs）上做细粒度资源控制
- 将正在 review 中的 `memcg_bpf_ops` patches upstream 到内核

代码、数据和论文位于 [https://github.com/yunwei37/agentcgroup-paper](https://github.com/yunwei37/agentcgroup-paper)。
