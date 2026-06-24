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

## 标签化问题：一个 Open Challenge

语义 flamegraph 的核心技术挑战是把自然语言 prompt 映射为稳定、有意义的标签。这比 CPU profiling 从根本上更难：CPU profiler 的函数名是确定性的符号。我们有可用的方案，但不是已解决的方案，我们对局限性保持坦诚。

### 为什么标签化很难

看一个真实开发 session 中的 prompt：

```
"fix the 编译 error"          # 混合语言
"嗯"                          # 单字符确认
"ok"                          # 意图模糊
"继续"                        # 依赖上下文
"[Session continued...]"      # 系统生成
"看看 collector/src/main.rs"  # 检查请求
"为啥 cargo test 失败了"       # Debug 问题
```

这些 prompt 展现了让朴素分类方法失效的特性：

1. **多语言混合**：同一个 prompt 里有英文和中文，有时在同一句话里
2. **长度极端变化**：从 1 个字符到多段落的上下文恢复
3. **上下文依赖**：「继续」离开前文毫无意义
4. **隐式意图**：「嗯」可能是确认、认可或思考停顿
5. **系统噪声**：自动生成的 session 继续消息、工具输出、错误信息

没有单一方法能处理所有情况。我们目前提供三种后端，各有不同的 tradeoff：

### 当前方案

**Regex + Agent 迭代**：生产环境默认。规则如 `prompt:debug='(?i)fix|error|bug|broken|为啥'` 对 prompt 文本做模式匹配。工作流是迭代的：运行 agentpprof，观察未匹配样本，编写规则，重复直到覆盖率超过 95%。新项目通常需要 5-10 轮。

优势：确定性、可重复、快速、无外部依赖。规则可以版本控制并在 CI 中运行。

劣势：每个项目需要手动工作。规则对 prompt 风格变化脆弱。无法处理语义相似性（如「fix the bug」vs「resolve the issue」）。

**LLM 标签器**：通过 llama.cpp 本地推理，使用 grammar-constrained decoding 确保输出有效的单词。我们使用小模型（0.6B-3B 参数）配合激进缓存。

优势：处理语义相似性和多语言 prompt。不需要编写规则。

劣势：非确定性（同一 prompt 多次运行可能得到不同标签）。需要本地模型配置。标签质量取决于模型能力。我们的实验显示 3B 模型在 300 个片段上有 285 个完全稳定，意味着 5% 的 prompt 重复运行会得到不同标签。

**TF-IDF + K-Means 聚类**：无监督聚类发现自然分组。自动选择聚类数（5-25）并从聚类关键词生成标签名。

优势：不需要预定义类别。发现你没有预料到的结构。

劣势：聚类边界是任意的。标签名来自关键词，不是语义。需要事后解读。

### 我们不知道的

几个根本性问题仍然开放：

**标签充分性**：我们可以验证标签语法有效且跨运行稳定（我们的 R180 实验显示三种模型大小都产出 900/900 语法有效输出）。但我们没有证据表明一词标签能捕获足够的语义信息供人类理解。「debug」可能把 bug 修复、错误调查和性能调试混在一起，而这些有不同的成本含义。

**跨项目迁移**：为一个项目开发的规则可能不能迁移到另一个项目。Rust 系统项目和 React 前端项目有不同的 prompt 模式。我们还不知道不同项目类型之间有多少规则重叠。

**最优粒度**：「code review」应该是一个标签，还是应该拆成 `review:style`、`review:logic`、`review:security`？更细的粒度保留信息但碎片化 flamegraph。我们没有原则性的方法来选择。

**多语言归一化**：「Fix the bug」和「修一下这个 bug」可能应该得到相同的标签，但 regex 规则无法表达这一点。LLM 标签器可以，但有稳定性 tradeoff。

### 为什么我们还是发布了

尽管有这些局限，agentpprof 在实践中是有用的。关键洞察是：完美标签不是有用聚合的必要条件。即使有 20% 未匹配 prompt 和不完美的标签边界，flamegraph 仍然揭示了之前不可见的结构：哪些活动类别占主导，token 消耗如何分布在意图类型之间，哪些 prompt 触发最多工具调用。

目标不是 ground-truth 分类，而是可操作的可见性。如果 flamegraph 显示「review」消耗 40% 的 token，什么算「review」的精确边界没那么重要，重要的是知道 review 类活动是主要成本驱动因素。

我们正在积极推进：
- LLM 辅助规则生成（模型从未匹配样本中提出规则）
- 基于 embedding 的相似度做多语言归一化
- 标签充分性的人工评估（目前我们的证据基础中缺失）

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

## 从可见性到行动：更难的问题

生成 flamegraph 是容易的部分。更难的问题是：你拿它做什么？

CPU profiler 导向明确的行动：找到热点函数，优化算法，减少分配。但 agent 成本 profile 不同：

- 你不会因为 code review 消耗 40% token 就停止代码审查
- 你不会因为 debug 昂贵就跳过 debug
- Flamegraph 显示预算去向，但不显示为什么去那里或如何减少

可操作的洞察需要更深入：

1. **类别内分析**：Review 消耗 40% token，但这是因为重复审查同一文件？不必要的宽上下文窗口？冗长的 review prompt？Flamegraph 显示类别；理解原因需要检查单个 session。

2. **工作流模式检测**：continuation prompt（`prompt:continue`）频繁出现可能表明任务应该在前期有更好的结构。高 `prompt:unmatched` 率可能表明 prompt 风格需要标准化。

3. **跨 session 比较**：这个月的 token 分布和上个月不同吗？工作流变化增加了 debug 成本吗？趋势分析需要基线比较。

我们正在将 agentpprof 与交互分析结合，产出推荐具体改变的报告：防止重复文件审查的 CLAUDE.md 规则、减少上下文开销的 prompt 模板、最小化 continuation churn 的工作流重构。

## 当前局限

**Agent 覆盖**：目前只读取 Codex 和 Claude Code 本地 trace。Gemini、Cursor 和其他 agent 需要通过 `agent-session` crate 扩展解析器。

**标签化**：如上所述，语义标签仍然是一个 open challenge。需要项目特定的规则，我们还没有证据表明一词标签语义充分。

**验证**：我们有机制证据（flamegraph 正确按标签聚合）但没有用户证据（开发者用这个视图做出更好决策）。后者需要我们尚未进行的用户研究。

**成本归因**：Token 数量来自 agent 报告的 usage，可能不反映实际账单（缓存 token、批量折扣、模型特定定价）。Flamegraph 显示相对分布，不是美元金额。

---

agentpprof 是开源的，属于 [AgentSight 项目](https://github.com/eunomia-bpf/agentsight)。欢迎贡献和反馈。
