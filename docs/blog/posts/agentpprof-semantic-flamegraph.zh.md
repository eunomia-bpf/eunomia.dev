---
date: 2026-06-24
slug: agentpprof-semantic-flamegraph
description: 你的 AI agent 这个月花了 $3000。哪些活动消耗了这些预算？agentpprof 将 flamegraph 范式应用于 AI agent trace，把自然语言 prompt 映射为语义标签，像 CPU profiler 一样聚合。本文解释为什么现有可观测工具无法回答预算归因问题，以及语义 flamegraph 如何为 agent 工作负载恢复聚合能力。
---

# 用语义 Flamegraph 分析 AI Agent：预算花在哪了？

月底账单显示 agent 花了 $3000。哪些类型的工作消耗了这些预算？代码审查占多少、debug 占多少、文档生成占多少？这个问题看似简单，但现有的 agent 可观测性工具都无法直接回答。

[agentpprof](https://github.com/eunomia-bpf/agentsight) 正是为回答这类问题而设计的分析工具。它读取本地 agent 的 trace 历史，按语义意图将 prompt 和工具调用聚合成 flamegraph：宽度代表 token 消耗、执行时间或操作次数的占比。一眼就能看出预算花在了哪类工作上。它是 [AgentSight](https://github.com/eunomia-bpf/agentsight) 项目的一部分，该项目提供基于 eBPF 的 AI agent 行为可观测性。

<!-- more -->

## 聚合问题

LangSmith、Langfuse、Phoenix 这类 LLM 可观测性平台能展示每次调用的 token 数和 latency，但当你有 80000 次调用时，它们只能按时间戳排列成 timeline。你可以逐条检查「这次调用花了 500 tokens」，但无法回答「审查类任务总共花了多少」。这些工具的设计目标是单次 trace 调试：timeline view 帮你定位 14:03 那个失败的 span，span tree 展示调用层级，waterfall chart 显示并行度。它们在回答「发生了什么」这个问题上表现出色，但对于「预算花在哪类工作上」这种聚合问题，逐条检查 80000 个 span 显然行不通。

Datadog 和 Laminar 开始尝试语义分类。Datadog 用 topic clustering 对用户消息做聚类，Laminar 用 Signals 从 trace 中提取结构化事件。这是正确的方向，但它们的聚类主要针对用户输入的分布，而不是产生「宽度代表预算占比」的聚合视图。你能看到「30% 的用户在问代码问题」，但看不到「代码审查消耗了 40% 的 token 预算」。

CPU profiler 早就解决了类似的聚合问题。Flamegraph 把百万次函数调用压缩成一张图，宽度代表时间占比。调用栈表示事件所属的上下文，对同一函数的重复调用会合并成更宽的条带。这之所以有效，是因为函数名是确定性的：相同的代码路径产生相同的调用栈，相同的调用栈可以直接合并。

Agent trace 打破了这个假设。Prompt 是自然语言：非确定性的、长度可变的、多语言的、往往还是对话式的。「Fix the bug」和「修一下这个 error」表达相同的意图，但字符串完全不同。如果直接用原始 prompt 文本作为 frame 标签，flamegraph 会宽得无法阅读，每个 prompt 都是独立的一条，失去了聚合的意义。而且原始 prompt 往往包含敏感信息，也不适合分享。

## 语义 Flamegraph：恢复聚合能力

agentpprof 通过引入语义标签来恢复聚合能力：将自由格式的 prompt 映射为简短、稳定的标签，如 `debug`、`review`、`paper` 或 `docs`。标签化之后，prompt 的行为就像函数名一样，重复的活动会合并，flamegraph 变得可读。

Flamegraph 的价值不只是聚合，还有堆栈表达因果关联。传统 CPU flamegraph 的堆栈是函数调用链：`main → parse → tokenize`，表示 tokenize 是被 parse 调用的，parse 是被 main 调用的。语义 flamegraph 的堆栈是 agent 行为的因果链：`prompt:debug → call:llm/analysis → tool:bash → file:src/main.rs`，表示这次文件修改是由 bash 工具触发的，bash 是 LLM 决定调用的，LLM 是在响应一个 debug 类型的 prompt。

| | 传统 CPU Flamegraph | 语义 Flamegraph |
| --- | --- | --- |
| **堆栈含义** | 函数调用链 | prompt → LLM → tool → effect 因果链 |
| **聚合方式** | 相同函数名合并 | 相同语义标签合并 |
| **宽度含义** | CPU 时间占比 | token / 时间 / 操作次数占比 |
| **回答问题** | 程序在哪里花 CPU | agent 在哪类工作上花预算 |

这种因果关联让你能从任意一层回溯或下钻：从某个文件被修改，追溯到是哪个工具、哪个 LLM 决策、哪个用户意图导致的；或者从某类 prompt 出发，看它触发了什么 LLM 调用、什么工具执行、什么系统效果。

## 多种视图，不同问题

agentpprof 对同一数据提供多种投影视图，每种回答不同的问题：

| 视图 | 宽度含义 | 主要回答的问题 |
| --- | ---: | --- |
| `tokens` | 报告的 token 数量（input/output/cache） | 哪些 prompt 消耗了最多的模型预算？ |
| `time` | 持续时间（秒） | 每个 prompt/活动花了多长时间？ |
| `files` | 文件/路径操作次数 | 哪些 prompt 触及了仓库的哪些部分？ |
| `network` | 网络/域名请求次数 | 哪些 prompt 联系了哪些域名？ |

从 `tokens` 视图开始定位成本热点，再用 `time` 追踪 wall-clock 时间去向，`files` 和 `network` 则适合安全审计场景。

## AgentSight 开发过程的真实示例

以下示例来自 AgentSight 项目自身的开发 trace（Claude Code）。它们展示了每个视图能提供什么洞察。

### Tokens 视图：模型预算花在哪了？

![Tokens flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-tokens.svg)

Token 分布显示代码审查（`prompt:review`）主导了模型预算，其次是 git 操作（`prompt:git`）、代码工作（`prompt:code`）、编辑（`prompt:edit`）和调试（`prompt:debug`）。通过堆栈可以追溯每类 prompt 触发了哪些 LLM 调用：`call:llm/usage` 表示 token 统计事件，`call:llm/code` 和 `call:llm/test` 表示代码相关响应，`call:llm/tool` 表示工具调用，`call:llm/edit` 表示修改响应。

### Time 视图：Wall-clock 时间花在哪了？

![Time flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-time.svg)

Wall-clock 时间分布与 token 消耗相似：review（`prompt:review`）领先，其次是 git、edit、docs 和 code 类 prompt。continuation prompt（`prompt:continue`）频繁出现，反映了复杂任务需要多轮后续交流的工作流模式。`prompt:inspect` 捕获了迭代开发中常见的「看一下」类请求。

### Files 视图：代码库的哪些部分被触及了？

![Files flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-files.svg)

文件访问模式显示 `collector/src/`（Rust 代码库）和 `collector/Cargo.toml` 活动频繁，与开发工作一致。外部路径（`external/tmp`、`external/home`、`external/codex`）也频繁出现，反映了工具调用触及临时文件、home 目录配置和 Codex session 数据。Flamegraph 区分了读取和写入效果，揭示了项目路径和外部路径上检查与修改之间的平衡。

### Network 视图：联系了哪些外部服务？

![Network flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-network.svg)

相对于文件操作，网络活动很少，确认大部分开发工作在本地完成。被联系的域名包括 `anthropic.com`（模型推理）、`crates.io`（Rust 依赖）、`github.com`（版本控制）以及各种 localhost 端口（本地开发服务器）。上层 frame 中可见的进程链显示了哪些工具发起了网络请求，使网络活动可以归因到具体的 agent 操作。

## 标签化的挑战

把自然语言 prompt 映射为稳定的语义标签并不容易。同一个项目里的 prompt 可能混合多种语言（「fix the 编译 error」），长度从单个字符（「嗯」、「ok」）到长段落不等，还有很多孤立看来没有意义的片段（「继续」、「好」、系统生成的上下文恢复消息）。

agentpprof 提供了一个可插拔的标签器框架，支持多种后端：

| 后端 | 方法 | 适用场景 |
| --- | --- | --- |
| Regex + Agent 迭代 | 正则匹配，由 AI agent 观察样本并迭代优化规则 | 生产环境、CI、可重复分析 |
| LLM 标签器 | 本地 LLM 推理（llama.cpp） | 复杂 prompt、初始规则发现 |
| Python 聚类 | TF-IDF + K-Means 无监督聚类 | 探索性分析、发现自然分组 |

推荐的工作流是让 AI agent 观察实际的 prompt 样本，不断迭代 regex 规则直到 unmatched 率降到 5% 以下。这个迭代过程通常需要 5-10 轮，最终产出的规则集是确定性的、可重复的，适合提交到版本控制并在 CI 中使用。

默认不包含内置规则，所有 prompt 会被标记为 `unmatched`。这是有意的设计选择：通用规则很难匹配你项目的实际 prompt 分布，盲目应用反而会产生误导性的聚合结果。

## 默认保护隐私

本地 agent 历史可能包含 prompt、工具输出、路径、命令、仓库名称和模型响应。agentpprof 默认采取保守策略：

- SVG、pprof 和 folded 输出只包含调用栈标签和权重，不包含原始 prompt 或模型响应。
- 所选项目根目录之外的绝对路径会被归类到稳定的桶中，如 `external/home`、`external/tmp`、`external/codex` 和 `external/claude`。
- 看起来私密的域名会被折叠，而不是暴露用户特定的主机名。

## AgentSight 的一部分

agentpprof 是 [AgentSight](https://github.com/eunomia-bpf/agentsight) 的离线分析组件，AgentSight 是一个基于 eBPF 的 AI agent 行为可观测性框架。AgentSight 通过 SSL/TLS 拦截和进程监控提供实时可见性，而 agentpprof 则对已记录的 agent trace 进行聚合分析。

典型工作流结合两者：

1. 用 `agentsight record` 录制 agent 活动
2. 用 `agentsight report` 生成摘要报告
3. 用 `agentpprof --view tokens` 分析 token 消耗
4. 用 `agentpprof --view files` 审计文件访问模式
5. 用 `agentpprof --view network` 检查网络目的地

安装和详细用法见 [AgentSight 仓库](https://github.com/eunomia-bpf/agentsight) 和 [agentpprof 文档](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md)。

## 局限性与未来工作

agentpprof 目前读取 Codex 和 Claude Code 的本地 trace 文件。其他 agent 可通过 `agent-session` 解析器扩展。语义标签方法需要针对项目开发规则，我们正在探索通过 LLM 辅助规则生成和基于聚类的发现来自动化这一过程。

更深层的问题是语义 flamegraph 是否能带来可操作的洞察。知道「代码审查消耗了 40% 的 token」很有趣，但你能用这个信息做什么？我们正在将 agentpprof 与交互分析结合，产出不仅展示预算去向，还能推荐具体工作流或 CLAUDE.md 改进的报告。

---

agentpprof 是开源的，属于 [AgentSight 项目](https://github.com/eunomia-bpf/agentsight)。欢迎贡献和反馈。
