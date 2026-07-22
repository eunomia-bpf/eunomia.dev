---
date: 2026-06-24
slug: agentpprof-semantic-flamegraph
description: AI agent trace 会把预算热点藏在成千上万条 prompt 里，agentpprof 用语义 flamegraph 聚合意图、token、时间、文件和网络。
---

# AI Agent Trace 的语义 Flamegraph

AI Agent 跑完一个月，你知道 token 总量却不知道分布。代码审查花了多少、调试花了多少？工具重试是否显著、continuation prompt 是否占主导？这些问题关乎工作流优化，但传统可观测性工具无法直接回答。

[agentpprof](https://github.com/eunomia-bpf/agentsight) 解决这个问题：读取本地 agent trace 历史，按语义意图将 prompt 和工具调用聚合成 flamegraph。宽度表示 token 消耗、执行时间或操作次数占比，团队可以先识别主导类别，再下钻到具体会话。它是 [AgentSight](https://github.com/eunomia-bpf/agentsight) 的一部分，后者是基于 eBPF 的 AI agent 行为可观测性框架。

宽条带不一定代表浪费。代码审查占比高可能是必要工作，也可能是反复读取同一批文件、上下文窗口过宽，或几种原因同时存在。分析图提供的是调查起点，不是结论。

<!-- more -->

## 聚合问题

大多数 LLM 可观测性工具围绕单次执行调试构建。时间线定位失败 span，调用树展示层级，瀑布图揭示并行度；这些视图回答单条 trace 里发生了什么。跨会话分析是另一个问题：调用数达到成千上万后，需要稳定的类别来合并重复工作，并用 token、耗时或操作次数等有意义的权重来度量。

主题聚类和结构化 trace 提取可以将相似输入归组，但输入分布不是预算画像。许多 prompt 提到代码，并不能说明审查消耗了多少模型预算、调用了哪些工具、影响了哪些文件。agentpprof 关注的是带权重的跨会话聚合。

CPU profiler 解决过结构相似的问题。Flamegraph 将百万次函数调用压缩成一张图，宽度表示时间占比。调用栈提供上下文，同一函数的重复调用合并成更宽的条带。这之所以有效，是因为函数名是确定性的：相同代码路径产生相同调用栈，相同调用栈直接合并。

Agent trace 打破了这个假设。Prompt 是自然语言：非确定性、长度可变、多语言、常为对话式。「Fix the bug」和「修一下这个 error」表达相同意图，但字符串完全不同。用原始 prompt 文本作 frame 标签，flamegraph 会宽得无法阅读，因为每个 prompt 都是孤立条带，失去聚合意义。原始 prompt 还常含敏感信息，不宜分享。

## 语义 Flamegraph

agentpprof 通过将自由格式 prompt 映射为简短稳定的标签来恢复聚合能力：`debug`、`review`、`paper`、`docs`。标签化后，prompt 表现得像函数名；重复活动合并，flamegraph 变得可读。

Flamegraph 除了聚合还编码因果链。传统 CPU flamegraph 里，`main → parse → tokenize` 表示 tokenize 被 parse 调用，parse 被 main 调用。语义 flamegraph 里，`prompt:debug → call:llm/analysis → tool:bash → file:src/main.rs` 表示 bash 触发了文件修改，LLM 决定调用 bash，LLM 在响应 debug 类型的 prompt。

| | 传统 CPU Flamegraph | 语义 Flamegraph |
| --- | --- | --- |
| **堆栈含义** | 函数调用链 | prompt → LLM → tool → effect 因果链 |
| **聚合方式** | 相同函数名合并 | 相同语义标签合并 |
| **宽度含义** | CPU 时间占比 | token / 时间 / 操作次数占比 |
| **回答问题** | 程序在哪里花 CPU | agent 在哪类工作上花预算 |

这种结构支持双向导航。从某个被修改的文件出发，追溯导致它的工具、LLM 决策和用户意图；从某类 prompt 出发，查看它触发的 LLM 调用、工具执行和系统效果。

## 视图

agentpprof 对同一 trace 数据提供多种投影，各回答不同问题。

| 视图 | 宽度含义 | 主要回答的问题 |
| --- | ---: | --- |
| `tokens` | 报告的 token 数量（input/output/cache） | 哪些 prompt 消耗了最多的模型预算？ |
| `time` | 持续时间（秒） | 每个 prompt/活动花了多长时间？ |
| `files` | 文件/路径操作次数 | 哪些 prompt 触及了仓库的哪些部分？ |
| `network` | 网络/域名请求次数 | 哪些 prompt 联系了哪些域名？ |

`tokens` 定位成本热点；`time` 追踪 wall-clock 时间；`files` 和 `network` 支持安全审计。

### 快速开始

生成当前仓库近期 Codex 或 Claude Code trace 的浏览器可打开分析图：

```bash
agentpprof --project-root . --view tokens --tagger regex --preset -o tokens.svg
```

`--preset` 提供演示标签规则，不是生产分类体系。需要重复比较时，应显式传入 `--session-file`，并用版本化规则替换 preset。打开 SVG 后先找最宽的 prompt 类别，再检查其下的具体会话，然后再决定是否调整工作流。

## AgentSight 开发示例

以下分析图来自 AgentSight 自身的 Claude Code trace。它们是描述性的，不是受控基准；类别名称取决于该项目的标签规则。示例展示 trace 获得稳定标签后可以检查哪些问题。

### Tokens

![Tokens flamegraph](imgs/agentsight-tokens.svg)

代码审查（`prompt:review`）主导 token 消耗，其次是 git 操作、代码工作、编辑和调试。堆栈将每类 prompt 追溯到它触发的 LLM 调用：`call:llm/usage` 表示 token 统计事件，`call:llm/code` 和 `call:llm/test` 表示代码相关响应，`call:llm/tool` 表示工具调用，`call:llm/edit` 表示修改响应。

### Time

![Time flamegraph](imgs/agentsight-time.svg)

Wall-clock 时间大体跟随 token 消耗：review 领先，其次是 git、edit、docs 和 code。continuation prompt（`prompt:continue`）频繁出现，反映需要多轮后续交流的任务。`prompt:inspect` 捕获迭代开发中常见的「看一下」类请求。

### Files

![Files flamegraph](imgs/agentsight-files.svg)

文件访问集中在 `collector/src/` 和 `collector/Cargo.toml`，与活跃的 Rust 开发一致。外部路径（`external/tmp`、`external/home`、`external/codex`）反映工具调用触及临时文件、home 目录配置和 Codex session 数据。Flamegraph 区分读取和写入，显示检查与修改的平衡。

### Network

![Network flamegraph](imgs/agentsight-network.svg)

相对于文件操作，网络活动很少，大部分工作在本地完成。被联系的域名包括 `anthropic.com`（模型推理）、`crates.io`（Rust 依赖）、`github.com`（版本控制）和 localhost 端口（本地服务器）。上层 frame 的进程链显示哪些工具发起了各请求，使网络活动可归因到具体操作。

## 标签化

核心挑战是将自然语言 prompt 映射为稳定、有意义的标签。这比 CPU profiling 更难，因为函数名是确定性符号。项目有可用方案但非已解决方案；局限性值得明确说明。

### 为什么标签化困难

真实开发 session 中的 prompt：

```
"fix the 编译 error"          # 混合语言
"嗯"                          # 单字符确认
"ok"                          # 意图模糊
"继续"                        # 依赖上下文
"[Session continued...]"      # 系统生成
"看看 collector/src/main.rs"  # 检查请求
"为啥 cargo test 失败了"       # Debug 问题
```

这些特性让朴素分类失效：多语言混合（同一 prompt 甚至同一句话中英文混用）、长度极端变化（1 字符到多段落上下文恢复）、上下文依赖（「继续」离开前文毫无意义）、隐式意图（「嗯」可能是确认、认可或思考停顿）、系统噪声（自动生成的 session 继续消息、工具输出、错误信息）。

没有单一方法能处理所有情况。agentpprof 提供三种后端，各有不同 tradeoff。

### 方案

**Regex + 迭代** 是生产默认方案。规则如 `prompt:debug='(?i)fix|error|bug|broken|为啥'` 对 prompt 文本做模式匹配。工作流需迭代：运行 agentpprof，观察未匹配样本，编写规则，重复直到覆盖率超过 95%。新项目通常需要 5 到 10 轮。结果确定、可重复、快速、无依赖，适合版本控制和 CI。代价是每个项目都需维护；规则对 prompt 风格变化脆弱，无法处理语义相似表达（「fix the bug」和「resolve the issue」）。

**LLM 标签器** 通过 llama.cpp 本地推理，将每次结果约束为单个标签，缓存输出以复用。它处理相似性和多语言 prompt 不需要编写规则。代价是稳定性：同一 prompt 不同运行可能得到不同标签，本地模型需配置，标签质量依赖模型能力。需要可重复结果时应缓存标签，再将有用类别转成确定性规则。

**TF-IDF + K-Means 聚类** 用无监督方法发现自然分组，自动选择聚类数（5 到 25）并从聚类关键词生成标签名。不需预定义类别，可能揭示未预料到的结构。代价是可解释性：聚类边界任意，标签名来自关键词而非语义，结果需事后解读。

### 开放问题

**标签充分性。** 语法约束保证标签格式有效，但不能证明一个词保留了足以支持人类判断的语义。「debug」可能把 bug 修复、错误调查和性能调试混在一起，而这些活动有不同的成本含义。

**跨项目迁移。** 为一个项目开发的规则可能无法迁移到另一个项目。Rust 系统项目和 React 前端项目有不同的 prompt 模式；有多少重叠尚不清楚。

**最优粒度。**「code review」应该是一个标签，还是拆成 `review:style`、`review:logic`、`review:security`？更细的粒度保留信息但碎片化 flamegraph。

**多语言归一化。**「Fix the bug」和「修一下这个 bug」可能应该得到相同标签，但 regex 规则无法表达。LLM 标签器可以，有稳定性 tradeoff。

## AgentSight 集成

agentpprof 是 [AgentSight](https://github.com/eunomia-bpf/agentsight) 的离线分析组件，AgentSight 是基于 eBPF 的 AI agent 行为可观测性框架。AgentSight 通过 SSL/TLS 拦截和进程监控提供实时可见性；agentpprof 聚合已记录的 trace。

典型工作流结合两者：用 `sudo agentsight record -- claude` 录制 agent 活动，用 `agentsight report` 生成摘要报告，然后用 `agentpprof --view tokens`、`--view files` 或 `--view network` 分析。

安装和详细用法见 [AgentSight 仓库](https://github.com/eunomia-bpf/agentsight) 和 [agentpprof 文档](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md)。

## 从可见性到行动

生成 flamegraph 是容易的部分；决定拿它做什么更难。CPU 性能分析导向明确行动：找热点函数、优化算法、减少内存分配。Agent 成本画像不同：你不会仅因代码审查是最宽类别就停止审查，不会因 debug 昂贵就跳过 debug。Flamegraph 显示预算去向，不显示为什么或如何减少。

可操作的洞察需要继续下钻。类别内分析追问为什么 review 是最宽类别：重复审查同一文件、不必要的宽上下文窗口、冗长 prompt。工作流模式检测关注反复出现的交互形态：频繁的 continuation prompt 可能表明任务应该在前期有更好的结构；高 unmatched 率可能表明 prompt 风格需要标准化。跨 session 比较回答这个月的分布是否不同于上个月，某次工作流变化是否增加了 debug 成本。

正在将 agentpprof 与交互分析结合，产出推荐具体改变的报告：防止重复文件审查的 CLAUDE.md 规则、减少上下文开销的 prompt 模板、最小化 continuation churn 的工作流重构。

## 局限

**Agent 覆盖** 仅限 Codex 和 Claude Code 本地 trace。其他 agent 需通过 `agent-session` crate 扩展解析器。

**标签化** 仍是开放问题。需要项目特定规则，尚无证据表明一词标签语义充分。

**验证** 停留在机制证据（flamegraph 正确按标签聚合），而非用户证据（开发者用此视图做出更好决策）。后者需要尚未进行的用户研究。

**成本归因** 依赖 agent 报告的 usage，可能不反映实际账单；缓存 token、批量折扣和模型特定定价都会改变费用。Flamegraph 显示相对分布，不是美元金额。

---

agentpprof 是开源的，属于 [AgentSight 项目](https://github.com/eunomia-bpf/agentsight)。欢迎贡献和反馈。

## 参考资料

- [AgentSight 仓库](https://github.com/eunomia-bpf/agentsight)
- [agentpprof 文档](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md)
- [AgentSight：使用 eBPF 实现 AI Agent 系统级可观测性](https://arxiv.org/abs/2508.02736)
- [Brendan Gregg，Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- [Go 诊断文档：性能分析](https://go.dev/doc/diagnostics#profiling)
