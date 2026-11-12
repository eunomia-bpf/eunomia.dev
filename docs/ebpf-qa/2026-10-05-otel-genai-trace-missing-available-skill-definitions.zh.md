# 为什么 trace 看不出智能体当时有哪些技能可选，新的技能定义属性如何让这份列表可见？

现行语义约定只记录实际运行了哪个技能，不记录候选集里有什么。已合并的 `gen_ai.skill.name`、`gen_ai.skill.description`、`gen_ai.skill.source.uri`、`gen_ai.skill.resource.name` 属性挂在 `execute_tool` span 上，所以 trace 能看到的是智能体对它选中那个技能做了什么。看不到的是它没选什么、为什么没选：智能体跳过相关技能、或者抓了个相似但错误的技能时，trace 无法分辨这个技能是压根没被提供、提供时描述不好、还是输给了更相似的候选。新的 `gen_ai.skill.definitions` 属性（GenAI 语义约定仓库中开放的拉取请求 557）把候选列表本身放到内层 `invoke_agent` span 上，同一份 trace 就能回答"当时智能体可选的是什么"。

## 机制

技能遥测被刻意做成属性而非独立的 span 类型，因为工具名因框架而异：一个框架把技能生命周期拆成 `load_skill`、`load_skill_resource`、`run_skill_script` 三个工具，另一个框架对同样阶段起别的名字。已记录的是被解析出的技能——`name`、`description`、`source_uri`，以及被触碰的技能相对资源——挂在执行这次加载的那个 `execute_tool` span 上。由此留下的盲区恰恰是"候选里有什么"：只有被选中的技能可见。

新属性补的正是这一半。`gen_ai.skill.definitions` 是内层 `invoke_agent` span 上的一个 opt-in、development 稳定性级别的属性，在调用开始时记录。取值是技能定义数组。每个定义遵循 Agent Skills 规范——一个技能就是一个文件夹，其 `SKILL.md` 前言里 `name`（1 到 64 字符，小写字母数字与连字符）和 `description`（1 到 1024 字符）是必填的——因此定义携带的就是这一对，外加可选的 `source_uri`（镜像 `gen_ai.skill.source.uri`）、`compatibility`、`license`、`metadata`，以及试验性的 `allowed-tools` 字段。

显而易见的替代方案——复用 `gen_ai.tool.definitions`——并不合适。那个属性描述的是函数调用工具，身份里带参数 schema；技能没有参数 schema，它是指令加可选捆绑资源组成的文件夹，硬塞进工具形状等于虚构一个不存在的 schema。

这个属性要标准化的数据本来就有：编码智能体早把提供给模型的技能列表写进磁盘上的会话转录（有的转录在会话开头有一个带技能名与描述的 `skill_listing` 条目，有的在第一段 developer 消息里带一个"可用技能"块）。属性给这份列表一个 trace 里的标准位置，而不必让每个 trace UI 去刮各个智能体的转录。

## 验证与调试路径

1. 在 trace 里找内层 `invoke_agent` span 的 `gen_ai.skill.definitions`。缺失只说明插桩没有发射它或没开 opt-in，不能当"没有可用技能"的证据。
2. 看名为 `load_skill` 的 `execute_tool` span 上的 `gen_ai.skill.*` 属性：那是"实际跑了什么"的一半，含经 `source.uri` 记录的技能来源。
3. 诊断靠两者对比。你期望的技能若不在定义列表里，是它从未被提供；在列表里，问题就落在描述质量或相似候选之间的模糊上——已记录的 `description` 值可以直接检查。
4. 属性存在时，用拉取请求里的 JSON schema 校验取值：数组；每个条目有匹配 `^[a-z0-9]+(-[a-z0-9]+)*$` 的 `name` 和非空的 `description`。注意语义：列表是调用开始时的快照，运行中途增删的技能不会反映。

## 局限

- 截至本次运行，该拉取请求尚未合并；属性不属于任何已发布的约定，development 稳定性意味着形态在发布前仍可能变动。
- 它是 opt-in，多数插桩不开启就不会发射。属性缺失不等于没有可用技能。
- 注册表给该属性打了可能敏感的标记：技能名与描述可能透露内部流程与数据。往 collector 发多少要自己定。
- 可选属性默认不填充，因为列表可能很大；`source_uri` 与 `compatibility` 需要插桩侧开启开关。
- 属性记录的是被提供的集合，不是正确的集合：它把"它当时被提供过吗"从猜测变成事实，但模糊描述只有在读已记录的描述时才成为可诊断的问题。

## 参考

- [semantic-conventions-genai 拉取请求 557](https://github.com/open-telemetry/semantic-conventions-genai/pull/557) — 在内层 `invoke_agent` span 上新增 `gen_ai.skill.definitions`，含 schema、参考场景与动机。
- [semantic-conventions-genai 拉取请求 498](https://github.com/open-telemetry/semantic-conventions-genai/pull/498) — 已于 2026-09-29 合并：`execute_tool` span 上的 `gen_ai.skill.name`、`gen_ai.skill.description`、`gen_ai.skill.source.uri`、`gen_ai.skill.resource.name` 属性。
- [semantic-conventions-genai — GenAI 属性注册表](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md) — 已发布的 `gen_ai.skill.*` 与 `gen_ai.tool.definitions` 定义，新属性与之对照。
- [semantic-conventions-genai 拉取请求 557 — skill-definitions JSON schema](https://github.com/open-telemetry/semantic-conventions-genai/blob/0e098d9bef6118b36819f83ccea438fa8dba09b0/model/gen-ai/gen-ai-skill-definitions.json) — `SkillDefinition` 形状：必填 `name` 与 `description`，可选 `source_uri`、`compatibility`、`license`、`metadata`、`allowed-tools`。
- [Agent Skills 规范](https://agentskills.io/specification) — 技能是带 `SKILL.md` 的文件夹，其前言的 `name` 与 `description` 必填。

## 当日社区讨论

选取的问题来自一个 opt-in 存档：一条讨论公告了一个记录"智能体调用开始时有哪些 Agent Skills 可用"的拉取请求，使 trace 能区分"技能从未被提供"与"提供过但描述不佳或输给了相似候选"——诉求是对新属性、其 schema 及在智能体调用 span 上的位置的审阅。该讨论还提到编码智能体早已把候选技能列表写进会话转录，这正是标准 trace 属性可以不依赖新数据采集而可行的原因。

当日其他讨论：一条 OpenTelemetry eBPF 智能体关于 Kubernetes 缓存地址环境变量在 Helm 图渲染 Config v2 文档时被忽略的讨论——两天前发布问题的再次提交，现附一个 helm 图拉取请求、线内确认这是真实问题、以及把缓存地址手动写入 Kubernetes 增强器配置的手工绕行；已发布的答案已覆盖该问题，不重复发布。另一条讨论文高频 socket 层重试（每秒数千次并发）下、重 ring buffer 负载里内核态 drop 时延与用户态 context switch 的对比基准；没有公开一手资料与决定性边界可依，未发布。

本次运行的渠道覆盖：两个 opt-in 存档共 8 条消息，全部如上覆盖。visible-browser-only 来源（Discord、eunomia-bpf 与 sched-ext 社区、bpf 邮件列表、r/eBPF）本次未能审阅——没有可用的 visible-browser 会话——因此标记为未覆盖，而非平静。
