---
date: 2026-06-24
slug: agentpprof-semantic-flamegraph
description: AI agent trace 会把预算热点藏在成千上万条 prompt 里，agentpprof 用语义 flamegraph 聚合意图、token、时间、文件和网络。
---

# AI Agent 预算花到哪了：用语义 Flamegraph 诊断 Trace

一张 $3000 的 AI agent 账单不是诊断报告，它不会告诉团队钱是花在代码审查、调试、文档生成、工具重试，还是把同一个任务拖成多轮的 continuation prompt 上。真正的运营问题不只是 agent 花了多少钱，而是哪几类重复工作已经贵到值得重写流程。

[agentpprof](https://github.com/eunomia-bpf/agentsight) 读取本地 agent 的 trace 历史，按语义意图将 prompt 和工具调用聚合成 flamegraph。在这个视图里，宽度代表 token 消耗、执行时间或操作次数的占比，团队可以先看出哪些类别占主导，再下钻到背后的具体会话。它是 [AgentSight](https://github.com/eunomia-bpf/agentsight) 项目的一部分，该项目提供基于 eBPF 的 AI agent 行为可观测性。

最宽的条带不一定代表浪费。代码审查占比高，可能是这项工作本来就必要，也可能是 Agent 反复读取同一批文件、上下文窗口过宽，或者几种原因同时存在。这张分析图的作用是给团队一个可以继续核查的起点。

<!-- more -->

## 聚合问题

大多数 LLM 可观测性视图以单次执行为中心。时间线用来定位失败 span，调用树展示层级，瀑布图揭示并行度，它们适合回答一条 trace 里发生了什么。跨会话的问题需要另一种数据形态。调用数达到成千上万后，读者需要稳定的类别来合并重复工作，并用 token、耗时或系统操作次数作为权重。

主题聚类和结构化 trace 提取可以把相似输入放在一起，但输入分布还不是预算画像。很多 prompt 都提到代码，并不能说明审查到底消耗了多少模型预算、调用了哪些工具，或者影响了哪些文件。agentpprof 关注的是这种带权重的跨会话聚合。

CPU profiler 早就解决了类似的聚合问题。Flamegraph 把百万次函数调用压缩成一张图，宽度代表时间占比。调用栈表示事件所属的上下文，对同一函数的重复调用会合并成更宽的条带。这之所以有效，是因为函数名是确定性的，相同的代码路径会产生相同的调用栈，相同的调用栈可以直接合并。

Agent trace 打破了这个假设。Prompt 是自然语言，因此非确定性、长度可变、多语言，而且往往还是对话式的。「Fix the bug」和「修一下这个 error」表达相同的意图，但字符串完全不同。如果直接用原始 prompt 文本作为 frame 标签，flamegraph 会宽得无法阅读，每个 prompt 都是独立的一条，失去了聚合的意义。而且原始 prompt 往往包含敏感信息，也不适合分享。

## 语义 Flamegraph 恢复聚合能力

agentpprof 通过语义标签来恢复聚合能力，将自由格式的 prompt 映射为简短、稳定的标签，如 `debug`、`review`、`paper` 或 `docs`。标签化之后，prompt 的行为就像函数名一样，重复的活动会合并，flamegraph 变得可读。

Flamegraph 的价值不只是聚合，还有堆栈表达因果关联。传统 CPU flamegraph 的堆栈编码函数调用链，`main → parse → tokenize` 表示 tokenize 是被 parse 调用的，parse 是被 main 调用的。语义 flamegraph 的堆栈编码 agent 行为的因果链，`prompt:debug → call:llm/analysis → tool:bash → file:src/main.rs` 表示这次文件修改是由 bash 工具触发的，bash 是 LLM 决定调用的，LLM 是在响应一个 debug 类型的 prompt。

| | 传统 CPU Flamegraph | 语义 Flamegraph |
| --- | --- | --- |
| **堆栈含义** | 函数调用链 | prompt → LLM → tool → effect 因果链 |
| **聚合方式** | 相同函数名合并 | 相同语义标签合并 |
| **宽度含义** | CPU 时间占比 | token / 时间 / 操作次数占比 |
| **回答问题** | 程序在哪里花 CPU | agent 在哪类工作上花预算 |

这种因果关联让你能从任意一层回溯或下钻。从某个被修改的文件出发，可以追溯到导致它的工具、LLM 决策和用户意图。从某类 prompt 出发，可以看到它触发了哪些 LLM 调用、工具执行和系统效果。

## 多种视图，不同问题

agentpprof 对同一数据提供多种投影视图，每种视图回答不同的问题。

| 视图 | 宽度含义 | 主要回答的问题 |
| --- | ---: | --- |
| `tokens` | 报告的 token 数量（input/output/cache） | 哪些 prompt 消耗了最多的模型预算？ |
| `time` | 持续时间（秒） | 每个 prompt/活动花了多长时间？ |
| `files` | 文件/路径操作次数 | 哪些 prompt 触及了仓库的哪些部分？ |
| `network` | 网络/域名请求次数 | 哪些 prompt 联系了哪些域名？ |

从 `tokens` 视图开始定位成本热点，再用 `time` 追踪 wall-clock 时间去向，`files` 和 `network` 则适合安全审计场景。

### 先生成一份可重复的分析图

要快速查看当前仓库关联的近期 Codex 或 Claude Code trace，可以先生成浏览器可直接打开的 SVG：

```bash
agentpprof --project-root . --view tokens --tagger regex --preset -o tokens.svg
```

`--preset` 只适合快速试用，不是生产环境的分类体系。需要重复比较时，应显式传入 `--session-file`，并把 `--preset` 换成项目内版本化的规则。打开 SVG 后，先找最宽且有明确含义的 prompt 类别，再检查它下面的具体会话，最后才决定是否调整工作流。

## AgentSight 开发过程的真实示例

以下示例来自 AgentSight 项目自身的 Claude Code 开发 trace。它们是项目分析图，不是受控基准测试。类别名称取决于这个项目使用的标签规则，示例展示的是 trace 获得稳定标签后可以检查哪些问题。

### Tokens 视图显示模型预算花在哪

![Tokens flamegraph](imgs/agentsight-tokens.svg)

Token 分布显示代码审查（`prompt:review`）主导了模型预算，其次是 git 操作（`prompt:git`）、代码工作（`prompt:code`）、编辑（`prompt:edit`）和调试（`prompt:debug`）。通过堆栈可以追溯每类 prompt 触发了哪些 LLM 调用。在这里，`call:llm/usage` 表示 token 统计事件，`call:llm/code` 和 `call:llm/test` 表示代码相关响应，`call:llm/tool` 表示工具调用，`call:llm/edit` 表示修改响应。

### Time 视图显示 Wall-clock 时间花在哪

![Time flamegraph](imgs/agentsight-time.svg)

Wall-clock 时间分布与 token 消耗相似。review（`prompt:review`）领先，其次是 git、edit、docs 和 code 类 prompt。continuation prompt（`prompt:continue`）频繁出现，反映了复杂任务需要多轮后续交流的工作流模式。`prompt:inspect` 捕获了迭代开发中常见的「看一下」类请求。

### Files 视图显示哪些代码路径被触及

![Files flamegraph](imgs/agentsight-files.svg)

文件访问模式显示 `collector/src/`（Rust 代码库）和 `collector/Cargo.toml` 活动频繁，与开发工作一致。外部路径（`external/tmp`、`external/home`、`external/codex`）也频繁出现，反映了工具调用触及临时文件、home 目录配置和 Codex session 数据。Flamegraph 区分了读取和写入效果，揭示了项目路径和外部路径上检查与修改之间的平衡。

### Network 视图显示联系了哪些外部服务

![Network flamegraph](imgs/agentsight-network.svg)

相对于文件操作，网络活动很少，确认大部分开发工作在本地完成。被联系的域名包括 `anthropic.com`（模型推理）、`crates.io`（Rust 依赖）、`github.com`（版本控制）以及各种 localhost 端口（本地开发服务器）。上层 frame 中可见的进程链显示了哪些工具发起了网络请求，使网络活动可以归因到具体的 agent 操作。

## 稳定标签才是真正难点

语义 flamegraph 的核心技术挑战是把自然语言 prompt 映射为稳定、有意义的标签。这比 CPU profiling 从根本上更难，因为 CPU profiler 的函数名是确定性的符号。我们有可用的方案，但不是已解决的方案，我们对局限性保持坦诚。

### 为什么标签化很难

看一个真实开发 session 中的 prompt。

```
"fix the 编译 error"          # 混合语言
"嗯"                          # 单字符确认
"ok"                          # 意图模糊
"继续"                        # 依赖上下文
"[Session continued...]"      # 系统生成
"看看 collector/src/main.rs"  # 检查请求
"为啥 cargo test 失败了"       # Debug 问题
```

这些 prompt 展现了让朴素分类方法失效的特性。

1. **多语言混合** 会让同一个 prompt 里同时出现英文和中文，有时甚至在同一句话里
2. **长度极端变化** 覆盖从 1 个字符到多段落上下文恢复的跨度
3. **上下文依赖** 使「继续」离开前文后毫无意义
4. **隐式意图** 让「嗯」可能表示确认、认可或思考停顿
5. **系统噪声** 包括自动生成的 session 继续消息、工具输出和错误信息

没有单一方法能处理所有情况。我们目前提供三种后端，各有不同的 tradeoff。

### 当前方案

**Regex + Agent 迭代** 是生产环境默认方案。规则如 `prompt:debug='(?i)fix|error|bug|broken|为啥'` 对 prompt 文本做模式匹配。工作流需要迭代，先运行 agentpprof，观察未匹配样本，再编写规则并重复这个过程，直到覆盖率超过 95%。新项目通常需要 5-10 轮。

它的优势是运行可预测。规则确定、可重复、快速、无外部依赖，而且可以版本控制并在 CI 中运行。

代价是每个项目都需要维护。规则对 prompt 风格变化脆弱，也无法处理「fix the bug」和「resolve the issue」这类语义相似表达。

**LLM 标签器** 通过 llama.cpp 在本地推理，把每次结果约束为一个标签，并缓存输出以便复用。

它的优势是语义覆盖更好，可以处理语义相似性和多语言 prompt，也不需要编写规则。

代价是稳定性和运行配置。同一个 prompt 在不同运行中可能得到不同标签，本地模型需要单独配置，标签质量也依赖模型能力。需要可重复结果时应缓存标签，再把有用类别逐步转成确定性规则。

**TF-IDF + K-Means 聚类** 用无监督聚类发现自然分组，自动选择聚类数（5-25）并从聚类关键词生成标签名。

它的优势是发现能力，不需要预定义类别，也可能揭示你没有预料到的结构。

代价是可解释性较弱。聚类边界是任意的，标签名来自关键词而不是语义，结果仍然需要事后解读。

### 我们不知道的

几个根本性问题仍然开放。

**标签充分性** 仍未被证明。语法约束可以保证标签格式有效，却不能证明一个词保留了足够多、足以支持人类判断的语义。「debug」可能把 bug 修复、错误调查和性能调试混在一起，而这些活动有不同的成本含义。

**跨项目迁移** 仍不清楚。为一个项目开发的规则可能不能迁移到另一个项目。Rust 系统项目和 React 前端项目有不同的 prompt 模式。我们还不知道不同项目类型之间有多少规则重叠。

**最优粒度** 还没有原则性的答案。「code review」应该是一个标签，还是应该拆成 `review:style`、`review:logic`、`review:security`？更细的粒度保留信息但碎片化 flamegraph。

**多语言归一化** 仍然困难。「Fix the bug」和「修一下这个 bug」可能应该得到相同的标签，但 regex 规则无法表达这一点。LLM 标签器可以，但有稳定性 tradeoff。

### 为什么我们还是发布了

尽管有这些局限，标签不必完美才能检验聚合是否有用。Flamegraph 仍可以显示哪些已标记活动占主导、token 如何分布，以及哪些 prompt 类别触发了最多工具调用，同时把无法确定的片段明确保留为 `unmatched`。

目标不是 ground-truth 分类，而是可操作的可见性。如果 flamegraph 显示「review」消耗 40% 的 token，什么算「review」的精确边界没那么重要，重要的是知道 review 类活动是主要成本驱动因素。

当前工作集中在三个方向。
- LLM 辅助规则生成（模型从未匹配样本中提出规则）
- 基于 embedding 的相似度做多语言归一化
- 标签充分性的人工评估（目前我们的证据基础中缺失）

## 默认保护隐私

本地 agent 历史可能包含 prompt、工具输出、路径、命令、仓库名称和模型响应。agentpprof 默认采取保守策略。

- SVG、pprof 和 folded 输出只包含调用栈标签和权重，不包含原始 prompt 或模型响应。
- 所选项目根目录之外的绝对路径会被归类到稳定的桶中，如 `external/home`、`external/tmp`、`external/codex` 和 `external/claude`。
- 看起来私密的域名会被折叠，而不是暴露用户特定的主机名。

## AgentSight 的一部分

agentpprof 是 [AgentSight](https://github.com/eunomia-bpf/agentsight) 的离线分析组件，AgentSight 是一个基于 eBPF 的 AI agent 行为可观测性框架。AgentSight 通过 SSL/TLS 拦截和进程监控提供实时可见性，而 agentpprof 则对已记录的 agent trace 进行聚合分析。

典型工作流结合两者。

1. 用 `sudo agentsight record -- claude` 录制 agent 活动
2. 用 `agentsight report` 生成摘要报告
3. 用 `agentpprof --view tokens` 分析 token 消耗
4. 用 `agentpprof --view files` 审计文件访问模式
5. 用 `agentpprof --view network` 检查网络目的地

安装和详细用法见 [AgentSight 仓库](https://github.com/eunomia-bpf/agentsight) 和 [agentpprof 文档](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md)。

## 从可见性到行动是更难的问题

生成 flamegraph 是容易的部分，更难的问题是你拿它做什么。

CPU 性能分析通常导向明确行动，例如找到热点函数、优化算法或减少内存分配。Agent 成本画像却不同。

- 你不会仅仅因为代码审查是最宽的 token 类别就停止审查
- 你不会因为 debug 昂贵就跳过 debug
- Flamegraph 显示预算去向，但不显示为什么去那里或如何减少

可操作的洞察需要继续下钻。

1. **类别内分析** 要追问为什么 review 是最宽的 token 类别。原因可能是重复审查同一文件、不必要的宽上下文窗口，也可能是冗长的 review prompt。Flamegraph 显示类别，理解原因仍需检查具体 session。

2. **工作流模式检测** 关注反复出现的交互形态。continuation prompt（`prompt:continue`）频繁出现可能表明任务应该在前期有更好的结构，高 `prompt:unmatched` 率可能表明 prompt 风格需要标准化。

3. **跨 session 比较** 要回答这个月的 token 分布是否不同于上个月，以及某次工作流变化是否增加了 debug 成本。趋势分析需要基线比较。

我们正在将 agentpprof 与交互分析结合，产出推荐具体改变的报告，例如防止重复文件审查的 CLAUDE.md 规则、减少上下文开销的 prompt 模板，以及最小化 continuation churn 的工作流重构。

## 当前局限

**Agent 覆盖** 目前只包括 Codex 和 Claude Code 本地 trace。Gemini、Cursor 和其他 agent 需要通过 `agent-session` crate 扩展解析器。

**标签化** 仍然是一个 open challenge。需要项目特定的规则，我们还没有证据表明一词标签语义充分。

**验证** 目前停留在机制证据上，说明 flamegraph 可以正确按标签聚合，但还没有用户证据证明开发者能用这个视图做出更好决策。后者需要我们尚未进行的用户研究。

**成本归因** 依赖 agent 报告的 usage，可能不反映实际账单，因为缓存 token、批量折扣和模型特定定价都会改变费用。Flamegraph 显示相对分布，不是美元金额。

---

agentpprof 是开源的，属于 [AgentSight 项目](https://github.com/eunomia-bpf/agentsight)。欢迎贡献和反馈。

## 参考资料

- [AgentSight 仓库](https://github.com/eunomia-bpf/agentsight)
- [agentpprof 文档](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md)
- [AgentSight：使用 eBPF 实现 AI Agent 系统级可观测性](https://arxiv.org/abs/2508.02736)
- [Brendan Gregg，Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- [Go 诊断文档：性能分析](https://go.dev/doc/diagnostics#profiling)
