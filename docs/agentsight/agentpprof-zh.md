# agentpprof: 用语义 flamegraph 分析 AI agent

月底账单显示 agent 花了 $3000。哪些类型的工作消耗了这些预算？代码审查占多少、debug 占多少、文档生成占多少？这个问题看似简单，但现有的 agent 可观测性工具都无法直接回答。

`agentpprof` 正是为回答这类问题而设计的分析工具。它读取本地 agent 的 trace 历史，按语义意图将 prompt 和工具调用聚合成 flamegraph：宽度代表 token 消耗、执行时间或操作次数的占比。一眼就能看出预算花在了哪类工作上。当前支持 Codex 和 Claude Code 的本地 trace 文件，其他 agent 可通过 `agent-session` 解析器扩展。

## 现有工具的局限

LangSmith、Langfuse、Phoenix 这类 LLM 可观测性平台能展示每次调用的 token 数和 latency，但当你有 80000 次调用时，它们只能按时间戳排列成 timeline。你可以逐条检查「这次调用花了 500 tokens」，但无法回答「审查类任务总共花了多少」。这些工具的设计目标是单次 trace 调试：timeline view 帮你定位 14:03 那个失败的 span，span tree 展示调用层级，waterfall chart 显示并行度。它们在回答「发生了什么」这个问题上表现出色，但对于「预算花在哪类工作上」这种聚合问题，逐条检查 80000 个 span 显然行不通。

Datadog 和 Laminar 开始尝试语义分类。Datadog 用 topic clustering 对用户消息做聚类，Laminar 用 Signals 从 trace 中提取结构化事件。这是正确的方向，但它们的聚类主要针对用户输入的分布，而不是产生「宽度代表预算占比」的聚合视图。你能看到「30% 的用户在问代码问题」，但看不到「代码审查消耗了 40% 的 token 预算」。

CPU profiler 早就解决了类似的聚合问题。Flamegraph 把百万次函数调用压缩成一张图，宽度代表时间占比。调用栈表示事件所属的上下文，对同一函数的重复调用会合并成更宽的条带。这之所以有效，是因为函数名是**确定性的**：相同的代码路径产生相同的调用栈，相同的调用栈可以直接合并。

Agent trace 打破了这个假设。Prompt 是自然语言：非确定性的、长度可变的、多语言的、往往还是对话式的。「Fix the bug」和「修一下这个 error」表达相同的意图，但字符串完全不同。如果直接用原始 prompt 文本作为 frame 标签，flamegraph 会宽得无法阅读，每个 prompt 都是独立的一条，失去了聚合的意义。而且原始 prompt 往往包含敏感信息，也不适合分享。

## 语义 flamegraph

`agentpprof` 通过引入**语义标签**来恢复聚合能力：将自由格式的 prompt 映射为简短、稳定的标签，如 `debug`、`review`、`paper` 或 `misc`。标签化之后，prompt 的行为就像函数名一样，重复的活动会合并，flamegraph 变得可读。

Flamegraph 的价值不只是聚合，还有**堆栈表达因果关联**。传统 CPU flamegraph 的堆栈是函数调用链：`main → parse → tokenize`，表示 tokenize 是被 parse 调用的，parse 是被 main 调用的。语义 flamegraph 的堆栈是 agent 行为的因果链：`prompt:debug → call:llm/analysis → tool:bash → file:src/main.rs`，表示这次文件修改是由 bash 工具触发的，bash 是 LLM 决定调用的，LLM 是在响应一个 debug 类型的 prompt。

| | 传统 CPU Flamegraph | 语义 Flamegraph |
| --- | --- | --- |
| **堆栈含义** | 函数调用链 | prompt → LLM → tool → effect 因果链 |
| **聚合方式** | 相同函数名合并 | 相同语义标签合并 |
| **宽度含义** | CPU 时间占比 | token / 时间 / 操作次数占比 |
| **回答问题** | 程序在哪里花 CPU | agent 在哪类工作上花预算 |

这种因果关联让你能从任意一层回溯或下钻：从某个文件被修改，追溯到是哪个工具、哪个 LLM 决策、哪个用户意图导致的；或者从某类 prompt 出发，看它触发了什么 LLM 调用、什么工具执行、什么系统效果。

`agentpprof` 对同一数据提供多种投影视图，每种回答不同的问题：

| 视图 | 宽度含义 | 主要回答的问题 |
| --- | ---: | --- |
| `tokens` | 报告的 token 数量（input/output/cache） | 哪些 prompt 消耗了最多的模型预算？ |
| `time` | 持续时间（秒） | 每个 prompt/活动花了多长时间？ |
| `files` | 文件/路径操作次数 | 哪些 prompt 触及了仓库的哪些部分？ |
| `network` | 网络/域名请求次数 | 哪些 prompt 联系了哪些域名？ |

从 `tokens` 视图开始定位成本热点，再用 `time` 追踪 wall-clock 时间去向，`files` 和 `network` 则适合安全审计场景。

## Flamegraph 示例

以下示例来自 AgentSight 项目自身的开发 trace（Claude Code）。它们展示了每个视图能提供什么洞察。

### Tokens 视图

**问题：** 哪些活动消耗了最多的模型预算？

![Tokens flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-tokens.svg)

Token 分布显示代码审查（`prompt:review`）主导了模型预算，其次是讨论（`prompt:discuss`）、查询（`prompt:query`）和文档（`prompt:docs`）。通过堆栈可以追溯每类 prompt 触发了哪些 LLM 调用：`call:llm/usage` 表示 token 统计事件，`call:llm/tool` 表示工具调用，`call:llm/edit`、`call:llm/test` 等表示具体的响应类型。

### Time 视图

**问题：** Wall-clock 时间花在了哪里？

![Time flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-time.svg)

Wall-clock 时间分布与 token 消耗相似，但讨论类（`prompt:discuss`）和查询类（`prompt:query`）在 time 视图中占比更高，说明这些交互式 prompt 涉及更多思考时间。continuation prompt（`prompt:continue`）频繁出现，反映了复杂任务需要多轮后续交流的工作流模式。

### Files 视图

**问题：** 代码库的哪些部分被触及了，以什么方式？

![Files flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-files.svg)

文件访问模式显示大部分活动集中在项目内（路径以 `repo/` 为前缀）。`collector/src/` 和 `docs/` 子树频繁出现，与开发和文档工作一致。Flamegraph 区分了读取和写入效果，揭示了检查与修改之间的平衡。外部路径（`external/home`、`external/tmp`）较少，确认 agent 在预期的文件系统边界内运行。

### Network 视图

**问题：** 联系了哪些外部服务？

![Network flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-network.svg)

相对于文件操作，网络活动很少，确认大部分开发工作在本地完成。被联系的域名（`github.com` 用于版本控制，`api.anthropic.com` 用于模型推理）都是预期中的。没有出现意外的第三方服务，从供应链安全角度来看这令人放心。上层 frame 中可见的进程链显示了哪些工具发起了网络请求，使网络活动可以归因到具体的 agent 操作。

生成脚本及标签规则见 `docs/flamegraph-example/agentsight.sh`。

## 标签化

把自然语言 prompt 映射为稳定的语义标签并不容易。同一个项目里的 prompt 可能混合多种语言（「fix the 编译 error」），长度从单个字符（「嗯」、「ok」）到长段落不等，还有很多孤立看来没有意义的片段（「继续」、「好」、系统生成的上下文恢复消息）。为了应对这些挑战，`agentpprof` 提供了一个可插拔的标签器框架，支持多种后端：

| 后端 | 方法 | 适用场景 |
| --- | --- | --- |
| Regex + Agent 迭代 | 正则匹配，由 AI agent 观察样本并迭代优化规则 | 生产环境、CI、可重复分析 |
| LLM 标签器 | 本地 LLM 推理（llama.cpp） | 复杂 prompt、初始规则发现 |
| Python 聚类 | TF-IDF + K-Means 无监督聚类 | 探索性分析、发现自然分组 |

### Regex 标签器与 Agent 迭代工作流

Regex 标签器是生产环境的默认选择，但它的使用方式和传统正则表达式不同。**你不需要手写所有规则**。正确的工作流是让 AI agent 观察实际的 prompt 样本，不断迭代规则直到 unmatched 率降到 5% 以下。

AgentSight 提供了 `agentpprof-flamegraph` skill，指导 agent 完成这个迭代过程：

1. 运行 `agentpprof`，观察 unmatched 率和样本 prompt
2. 根据样本提出新的 `--tag-rule` 规则
3. 重新运行，测量覆盖率
4. 重复直到 unmatched < 5%、分布合理（10-20 个类别，无单一类别 > 50%）

这个迭代过程通常需要 5-10 轮，每轮 1-2 分钟。最终产出的规则集是确定性的、可重复的，适合提交到版本控制并在 CI 中使用。

默认不包含内置规则，所有 prompt 会被标记为 `unmatched`。这是有意的设计选择：通用规则很难匹配你项目的实际 prompt 分布，盲目应用反而会产生误导性的聚合结果。

规则格式是 `KIND:TAG=REGEX`：

```bash
agentpprof -o tokens.svg \
  --tagger regex \
  --tag-rule prompt:review='(?i)review|diff|regression' \
  --tag-rule prompt:test='(?i)cargo test|pytest|unit test' \
  --tag-rule prompt:debug='(?i)fix|error|bug|broken'
```

`KIND` 可以是 `prompt`、`llm` 或 `all`。`TAG` 必须是 3-12 个字母的小写英文单词。规则按命令行顺序求值，第一个匹配生效。

快速测试时可以用 `--preset` 启用内置的演示规则：

```bash
agentpprof -o tokens.svg --tagger regex --preset
```

### LLM 标签器

对于复杂 prompt 或初始规则发现，可以用本地 LLM 生成标签。运行一个 llama.cpp 兼容的服务器：

```bash
llama-server -m /path/to/model.gguf --port 8080
agentpprof -o tokens.svg --tagger llm --llama-url http://127.0.0.1:8080
```

LLM 标签默认缓存在 `$XDG_CACHE_HOME/agentpprof/tags.json`。LLM 标签器的输出可以作为编写 regex 规则的参考：观察 LLM 产生了哪些类别，然后为每个类别写一条 regex 规则。

### Python 聚类后端（实验性）

对于探索性分析，可以用 Python 聚类后端发现 prompt 的自然分组。这个后端使用 TF-IDF 向量化和 K-Means 聚类，无需预定义规则：

```bash
# 导出 prompt
agentpprof --project-root . --format json -o prompts.json

# 聚类并生成标签缓存
python agentpprof/backend/python/cluster_tagger.py \
  --input prompts.json --output tags.json --show-info

# 使用标签缓存
agentpprof --project-root . --tag-cache tags.json -o flamegraph.svg
```

聚类后端会自动选择最优的聚类数（5-25），并根据每个聚类的关键词生成标签名。这对于理解「我的 prompt 分布里有哪些自然类别」很有用，可以作为编写 regex 规则的起点。

## 安装

发布后可通过 `cargo install agentpprof` 安装，也可以从 AgentSight GitHub release artifacts 下载预编译的二进制文件。发布流水线从同一个 release tag 构建并测试 `agentsight` 和 `agentpprof`。

从源码构建：

```bash
cargo run --manifest-path agentpprof/Cargo.toml -- --version
cargo run --manifest-path agentpprof/Cargo.toml -- -o agent.pb.gz
```

## 第一个 profile

为当前仓库生成 token profile：

```bash
agentpprof --project-root . --view tokens -o tokens.pb.gz
```

使用标准 Go 工具打开 pprof profile：

```bash
go tool pprof -top tokens.pb.gz
go tool pprof -http=:0 tokens.pb.gz
```

生成可在浏览器打开的 flamegraph：

```bash
agentpprof --project-root . --view tokens -o tokens.svg
```

未指定 `--format` 时，扩展名决定输出格式：

```bash
agentpprof -o tokens.pb.gz  --view tokens   # pprof protobuf, gzip 压缩
agentpprof -o time.folded   --view time     # folded stack 文本
agentpprof -o files.svg     --view files    # 独立 SVG flamegraph
agentpprof -o network.json  --view network  # 脱敏后的 JSON 摘要和调用栈
```

## 读取什么数据？

`agentpprof` 读取 agent 原生的本地 trace 历史。目前支持通过 `agent-session` crate 解析的 Codex 和 Claude Code JSONL 文件。它不加载 eBPF 探针、不需要 root 权限、不录制实时进程。它是 AgentSight 的离线分析端：用 `agentsight` 观察实时系统行为，用 `agentpprof` 聚合已记录的 agent trace。

默认情况下，它扫描与 `--project-root` 匹配的近期本地 trace：

```bash
agentpprof --project-root /path/to/repo --view tokens -o tokens.svg
```

对于可重复的分析，传入明确的 trace 文件：

```bash
agentpprof \
  --project-root /path/to/repo \
  --session-file ~/.codex/sessions/.../session.jsonl \
  --session-file ~/.claude/projects/.../session.jsonl \
  --view tokens \
  -o tokens.folded
```

有用的筛选器：

```bash
agentpprof -o tokens.svg --agent codex
agentpprof -o tokens.svg --session-id 019ec5
agentpprof -o tokens.svg --session-tag debug
agentpprof -o tokens.svg --prompt-tag review
```

## 调用栈模型

语义 flamegraph 的调用栈是一种投影而非字面意义的函数调用栈：下层 frame 提供上下文（project、agent、prompt 类型），上层 frame 描述被计数的活动，具体形状随视图而变。

`tokens` 视图以模型预算作为宽度：

```text
project:agentsight;agent:claude;session:profile;prompt:debug;call:llm/debug;model:claude-opus-4-6;kind:input 4200
project:agentsight;agent:claude;session:profile;prompt:debug;call:llm/debug;model:claude-opus-4-6;kind:output 980
project:agentsight;agent:claude;session:profile;prompt:debug;call:llm/debug;model:claude-opus-4-6;kind:cache 150000
```

`time` 视图以 wall-clock 持续时间（秒）作为宽度：

```text
project:agentsight;agent:claude;session:profile;prompt:debug;kind:llm 45
project:agentsight;agent:claude;session:profile;prompt:debug;kind:tool 12
project:agentsight;agent:claude;session:profile;prompt:debug;kind:prompt 2
```

`files` 视图以仓库区域作为主分支：

```text
project:agentsight;agent:codex;session:release;prompt:docs;path:docs/flamegraph;effect:write;status:ok 1
```

`network` 视图以域名为中心：

```text
project:agentsight;agent:codex;session:release;prompt:publish;domain:crates.io;process:cargo;status:ok 1
```

选择哪个视图取决于你的问题：tokens 用于成本分析，time 用于性能分析，files 用于评估影响范围，network 用于安全审计。

## 隐私与脱敏

本地 agent 历史可能包含 prompt、工具输出、路径、命令、仓库名称和模型响应。`agentpprof` 默认采取保守策略：

- SVG、pprof 和 folded 输出只包含调用栈标签和权重，不包含原始 prompt 或模型响应。
- JSON 输出会脱敏预览内容，除非设置了 `--include-previews`。
- 所选项目根目录之外的绝对路径会被归类到稳定的桶中，如 `external/home`、`external/tmp`、`external/codex` 和 `external/claude`。
- 看起来私密的域名会被折叠，而不是暴露用户特定的主机名。

需要可重复性时使用明确的 `--session-file` 输入。仅在私有调试或已脱敏的 trace 中使用 `--include-previews`。

## 配合 AgentSight 使用

`agentsight` 提供实时可见性（进程树、文件效果、网络目的地），`agentpprof` 提供聚合分析（成本热点、时间分布）。典型工作流是先用 `agentsight` 录制，再用 `agentpprof` 分析：

```bash
sudo agentsight record -- claude
agentsight report
agentpprof --project-root . --view tokens -o tokens.svg
agentpprof --project-root . --view time -o time.svg
agentpprof --project-root . --view files -o files.svg
```

## 故障排除

如果找不到 trace，传入明确的 `--session-file` 路径，并确认 trace 的 `cwd` 与 `--project-root` 匹配。

如果标签过于笼统，为项目添加几条 `--tag-rule`。不要试图让每个 prompt 都独一无二。好的标签应该保留有用的语义多样性，同时合并无意义的长尾碎片。

如果 pprof 输出可以打开但看起来不熟悉，请记住样本单位不是 CPU 时间。使用 `go tool pprof -top` 检查最宽的语义 frame，然后在需要完整调用栈形状时生成 SVG 或 folded 输出。

如果公开产物可能包含敏感信息，优先使用 SVG、folded 或 pprof 输出，不要传 `--include-previews`。
