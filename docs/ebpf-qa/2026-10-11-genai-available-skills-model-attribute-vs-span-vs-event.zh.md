# 为什么 GenAI 遥测里可用智能体技能列表应当是挂在 span 上的专属属性，而不是单独的 span 或 span 事件？span 的时长模型又为何排除掉 span 形式？

它应当作为 `invoke_agent` span 上的一个结构化属性来记录，而不是自己的 span，也不是 span 事件。span 是带时长边界的单元——有 start timestamp 与 end timestamp，用来覆盖一次操作，重试也要算进去；span 事件是带时间戳的"发生"，是那个 span 里某个时刻发生的事。而被提供的技能列表两者都不是：它是调用开始时能力面的一份即快照，自身没有时长，也没有可以时间戳化的"发生"，所以它是参考数据，按属性挂上去——这正是 `gen_ai.tool.definitions` 已经在用的形态，而开源的 `gen_ai.skill.definitions` 提案也沿用了同样的形状。

## 机制

trace 的数据模型是原因。一个 Span 封装了名字、context、parent、kind、start 时间戳、end 时间戳、属性、一组 link、一组带时间戳的事件、以及状态。GenAI spans 规范又给了语义意图：GenAI span SHOULD 覆盖操作的时长，从操作发起开始，到响应完整接收或操作因错误或取消而结束为止，自动重试折叠进同一个 span。因此 span 断言的是一段工作区间。而被提供的技能列表是调用开始时刻智能体能力面的快照；它没有终点、没有时长、也没有要度量的操作，把它单独包进一个 span 就等于伪造一对不存在的起止时间，并为每个调用多挂一个静态节点，去承载一份输入性质的数据而非子操作。

事件形式则因相反的原因不成立。事件是带时间戳的发生，按发生顺序记录；它建模的是"某事在 t 时刻发生"——某个技能被加载、某个工具被执行、一个流式 chunk 到达。可用技能列表不是"发生了的事"，而是对状态的描述。选事件形式就得给它一个任意时间戳，并断言一次没有发生的"发生"。它是数据，而 span 上的数据就是属性。

大体积结构化负载在本套规范里本来就落在属性上。`gen_ai.tool.definitions` 是一个 any 类型、Development 状态的属性，承载一份结构化的工具定义列表，后端支持时就按结构化形式记录，不支持时就记成 JSON 字符串。技能没有参数 schema，所以工具定义的那套 schema 不贴切；专门的补充就是 `gen_ai.skill.definitions`，记在 `invoke_agent` span 上，让 trace UI 能展示智能体当时究竟能选哪些技能，让平台按智能体所见的样子展示这些技能，并支持按技能搜索 trace。

## 验证与调试路径

1. 打开 OpenTelemetry trace API 规范的 Span 一节：Span 封装了 start 与 end 时间戳、属性，以及一组带时间戳的事件；GenAI spans 规范则写明 span SHOULD 覆盖操作时长。这一次读法就是"span 是有边界的操作，不是静态参考数据的容器"的原因。
2. 读 Add Events 一节：事件带有"被记录时刻"的时间戳，且事件 SHOULD 保持记录顺序——这是一组带时间戳发生的数据模型，而参考列表不属于此类。
3. 读 GenAI 属性注册表里 `gen_ai.tool.definitions` 的条目：它是一个 any 类型、Development 状态的属性，承载一份结构化列表——这是"把一组定义按 span 属性承载"的先例。
4. 读开源的 `gen_ai.skill.definitions` 提案：它在内部 `invoke_agent` span 上，以专属结构化属性记录可用技能列表，作为工具定义属性的补充；并注明面向函数调用参数建的工具 schema 无法描述一个技能。

## 局限

- 该定义属性仍是开源提案、Development 状态而非稳定，消费者暂时不能指望各 SDK 都提供；逐技能属性——`gen_ai.skill.name`、`gen_ai.skill.description`、`gen_ai.skill.resource.name`、`gen_ai.skill.source.uri`——已经作为属性存在，定义属性最终会加入这一族。
- 专属属性是挂在单个 span 上的结构化数据；它本身不会自动建立可搜索索引。按技能搜索取决于后端是否为属性值建索引，而不是取决于 trace 数据模型。
- 这份列表可能很大；是否附带它由规范的内容捕获 opt-in 决定，而在不支持结构化属性值的后端上，记成 JSON 字符串是回退方案。
- 事件只有在关心的事实是"加载/执行了哪个技能、何时发生"时才对——执行侧，已被工具与技能操作 span 覆盖，那与"被提供"列表是另一类信号。

## 参考

- [OpenTelemetry trace API 规范](https://opentelemetry.io/docs/specs/otel/trace/api/) —— Span 数据模型（start 与 end 时间戳、属性、一组带时间戳的事件），以及 Add Events 规则：事件是带时间戳、有序的发生。
- [semantic-conventions-genai 的 GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) —— GenAI span SHOULD 覆盖操作时长、`invoke_agent` 操作，以及把结构化内容记在 span 属性上的模式。
- [semantic-conventions-genai pull request 557](https://github.com/open-telemetry/semantic-conventions-genai/pull/557) —— 开源的 `gen_ai.skill.definitions` 属性，落在内部 `invoke_agent` span 上，记录调用开始时可用的技能，作为工具定义属性的补充。
- [semantic-conventions-genai 的 GenAI 属性注册表](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md) —— `gen_ai.tool.definitions` 作为承载结构化列表的 any 类型属性，以及已有的 `gen_ai.skill.*` 属性族。

## 当日社区讨论

本 run 审查的两个 opt-in 存档渠道（两个 CNCF OpenTelemetry 插桩渠道）共产出十三条消息。本页回答的正是其中一条：一位规范审阅者追问为何把可用技能列表建模成 span——"没有合理的时长——它是个时间点上的东西"，按这说法倒更像 span 事件；答复方则重新界定目标——不是要把系统 prompt 解析出一份来，而是一个专属结构化属性，去补充既有工具定义属性，好让平台按智能体所见的样子展示这些技能，并支持按技能搜索 trace。其余话题早已被往期回答覆盖——model-serving-signals 指标目录问题、`logger.enabled` 的 opt-in/opt-out 问题——而 Node.js GenAI 仓库问题本 run 没有可决出的技术边界，故未选取。

本 run 的渠道覆盖：两个 opt-in 存档渠道共提供十三条消息。可见浏览器专属的渠道——eunomia-bpf 与 sched-ext 的 Discord 服务器、bpf 邮件列表、以及 r/eBPF——本 run 没有可用的可见浏览器会话，因此标记为"未覆盖"，而非"安静"。
