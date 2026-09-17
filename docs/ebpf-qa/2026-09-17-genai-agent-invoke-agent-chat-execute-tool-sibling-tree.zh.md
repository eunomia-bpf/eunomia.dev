# GenAI 智能体 span 是否应把每次工具执行嵌套在请求它的模型调用下面，还是把模型调用与工具执行都作为同一个智能体 span 的兄弟节点？

**简短回答：** 保持兄弟节点即可。对于进程内智能体，每个客户回合一个 `invoke_agent` span，把模型调用（`chat`）与工具执行（`execute_tool`）都作为该 `invoke_agent` 的**直接子节点**，这就是约定的形状；工具执行与"请求它的那次模型调用"之间的对应关系，是靠 `gen_ai.tool.call.id` 这个属性来关联的，而不是靠嵌套。GenAI 智能体 span 约定里明确写着："由 plan 产生的工具或任务 span 通常是同一个 `invoke_agent` span 下的兄弟操作。" 把每个 `execute_tool` 嵌套到请求它的 `chat` 下面，是一种"因果树"的思路（只有当你能跨这一跳携带 W3C `traceparent` 时才值得做）；它不是必需项，"兄弟节点 + `gen_ai.tool.call.id`"也不是错的。

## 约定真正画出的那条边界

GenAI 语义约定把智能体 span 的词汇定义为 `invoke_agent`、`invoke_workflow`、`plan`、`chat`、`execute_tool`，每个都带一个 `gen_ai.operation.name`。形状之争落在 *Plan span* 一节里的一句话上：

> plan span 表示智能体在执行前制定策略的决策阶段。生成该 plan 的 LLM 调用应当是 plan span 的子节点，而**由 plan 产生的工具或任务 span 通常是同一个 `invoke_agent` span 下的兄弟操作。**

这就是决定性边界。"请求"工具的模型调用，和"执行"工具的 `execute_tool` span，是同一个 `invoke_agent` 下的兄弟，而不是父子对。把两者接起来的是 `gen_ai.tool.call.id` 属性（一个 `Recommended` 属性，"工具调用标识符"，用来把模型请求与兑现它的工具关联起来），所以这条关联承载在数据里，而不是承载在树结构里。

## 为什么 span kind 决定了你能建出哪种树

`invoke_agent` span 有两个变体，它们承载了形状：

- **`invoke_agent` INTERNAL** —— "同进程内的 GenAI 智能体调用"（例如进程内的 LangChain / CrewAI 风格智能体）。Span kind 为 `INTERNAL`。这是你自己拥有循环、自己发 `chat`/`execute_tool` 的情形，此时"单个 `invoke_agent` 下的兄弟树"正是你能控制的那棵树。
- **`invoke_agent` CLIENT** —— "跨远端服务的 GenAI 智能体调用"，约定把 OpenAI Assistants API 与 AWS Bedrock Agents 列为例子。Span kind 为 `CLIENT`。

CLIENT 变体是这棵树的极限：一旦智能体循环跑在供应商机器上的托管 harness 里，内部的 `chat`/`execute_tool` 树只有在 *harness 主动导出* 时才存在。通用 HTTP 客户端插桩仍然能看到的，是那些离开本盒子的出站 HTTP 交叉。所以"每个回合一个 `invoke_agent`，`chat`/`execute_tool` 作子节点"这个答案，干净地适用于进程内情形；对托管 harness 而言，内部树是供应商选择导出的东西。

## 关联到底怎么工作，以及要验证什么

因为这条关联是属性而不是父子链接，验证路径很简单：

1. **每个回合一个 `invoke_agent`，用会话做键。** 对进程内的支持型智能体，当"这个回合就是那次调用"时，每个客户回合一个 `invoke_agent` 是对的。用 `gen_ai.conversation.id` 把同一张工单跨回合绑在一起；除非确实有一个长时间运行的单次 run，否则不要把单个 `invoke_agent` 撑到整个多回合会话。
2. **`chat` 与 `execute_tool` 作为该 `invoke_agent` 的直接子节点。** 它们是兄弟。在"请求工具的模型调用"和"它产生的 `execute_tool`"两边都设 `gen_ai.tool.call.id`，这样即便树里没有那条因果边，读者也能把它们 join 起来。
3. **只有当你能把上下文带过去时才嵌套。** 把 `execute_tool` 嵌套到请求它的 `chat` 下面是"因果树"思路。只有当你能跨这一跳传播 W3C `traceparent`、让嵌套有意义时它才划算；如果不能，"兄弟 + `gen_ai.tool.call.id`"就是正确的、不算缺失的形状。
4. **确有规划时就检查 plan span。** 如果智能体做真正的任务分解，加一个 `plan` span（kind `INTERNAL`）；它的 LLM 调用是它的子节点，它产生的工具/任务 span 仍作为兄弟留在 `invoke_agent` 下。当无法可靠区分"规划"与普通推理时，省略 `plan`。

## 决定性的边界

约定树是"每次调用一个 `invoke_agent`，`chat` 与 `execute_tool` 为兄弟，靠 `gen_ai.tool.call.id` 关联"。把每个 `execute_tool` 嵌套到请求它的 `chat` 下面，是一种可选的因果细化，不是约定；只有当 W3C 上下文传播携带了这个嵌套时它才值得做。而且这棵树只在循环*属于你自己*、由你插桩时才成立：一旦循环搬进托管 harness，`chat`/`execute_tool` 这些子节点就是供应商选择导出的，通用 HTTP 插桩能重建的只有出站交叉。"兄弟形状 + 关联属性"不是缺失，而是文档化的默认。

## 参考资料

- [OpenTelemetry GenAI 语义约定：agent and framework spans（`invoke_agent` INTERNAL = "同进程内"；plan span 一句"由 plan 产生的工具或任务 span 通常是同一个 `invoke_agent` span 下的兄弟操作"）](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md)
- [OpenTelemetry GenAI 语义约定：GenAI spans（`chat` span kind `CLIENT` / `INTERNAL`；`execute_tool` span kind `INTERNAL`，span 名 `execute_tool {gen_ai.tool.name}`）](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [OpenTelemetry GenAI 语义约定（已发布的 `gen_ai.*` 属性注册表，含 `gen_ai.tool.call.id`）](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OpenTelemetry W3C Trace Context（跨因果一跳传播 `traceparent`）](https://www.w3.org/TR/trace-context/)
- [OpenAI Agents API 可观测性（公开 beta："新会话默认开启 tracing。公开 beta API 不暴露 tracing 配置或外部 trace 导出器。"）](https://developers.openai.com/api/docs/guides/agents-api/observability)

## 当日社区讨论

本监控窗口覆盖了两个白名单 Slack 归档（均为技术内容）以及两个仅可见浏览器的聊天工作区与公共邮件列表/论坛表面（后者本次未审阅）；技术内容来自那两个支持归档的工作区。

**GenAI 智能体 span 形状（即上文问题）。** 一位在用 TypeScript 构建小型开源支持型智能体的使用者，在写 agent 循环之前先按 GenAI 约定写好了 trace 层：一个回合、把模型调用导出为 `chat`、把工具执行导出为 `execute_tool`，模型调用与工具执行作为回合下的兄弟，并用 `gen_ai.tool.call.id` 关联。悬而未决的问题是：这是不是约定的形状，还是每一轮模型都应把它请求的工具执行嵌套起来。一位维护者对照当前 agent-spans 文档确认这棵树是约定：对进程内智能体，`invoke_agent` 是 `INTERNAL`，当回合"就是"那次调用时每个客户回合一个 `invoke_agent`，`chat`/`execute_tool` 作直接子节点，而"嵌套到请求它的 `chat` 下"这种因果树思路只有在能跨那一跳携带 W3C `traceparent` 时才划算。实用结论：保持"兄弟 + `gen_ai.tool.call.id`"；每个回合（而非每个会话）一个 `invoke_agent`；用 `gen_ai.conversation.id` 把工单跨回合串起来。

**托管 harness 与可观测性锁定。** 另一条线担忧 OpenAI Agents API 作为托管 harness：因为循环跑在供应商机器上，内部 `chat`/`execute_tool` 树只有在供应商导出时存在，而公开 beta 不暴露 tracing 配置或外部 trace 导出器——正是托管云服务把 o11y 数据留在自家体系内的通病。边界就是 CLIENT 版 `invoke_agent` 的情形：通用 HTTP 客户端插桩仍能抓到离开本盒子的出站交叉，但没法像进程内自发自管 span 的 SDK 那样重建完整内部 trace。一位使用者提出，把模型/工具调用当成一个普通的 HTTP 回执头（在一个本就普通的 span 上做供应商前缀映射），这样既有 HTTP 客户端插桩就能采到，而无需发明新的 `gen_ai.*` 属性。

**OBI 配置 v1 到 v2 迁移。** 一位使用者报告 `migrate` 命令报 "fields are outside the supported v1-to-v2 migration contract"，问是否应该做部分迁移。维护者说全有或全无是故意的（静默丢字段会产出"看起来合法但行为实质不同"的配置），但建议加一个显式的 `--allow-partial` / `--best-effort` 模式，把能迁移的迁成合法 v2、并报告被省略的字段，而不是把部分迁移做成默认。相关的 Helm chart 缺口（chart 的 `_helpers.tpl` 仍往 v2 配置 schema 里塞 v1 字段）被要求单独开 issue，附上 chart/OBI 版本、相关 values、渲染后的配置与校验错误。

**Node.js 哨兵性能（承接前一日）。** 关于注入 Node 代理 `async_hooks` 哨兵的成本归属，已推进到一个可评审的草稿 PR；一位使用者报告，把某服务排除出追踪后必须 rollout 重启 pod 才看到延迟回落——提示即便排除后，追踪进程仍可能挂在 pod 内。维护者指出，要 gate 的就是那个"每个回调的刷新"（如果它只是填充 trace 上下文 map，就应允许关掉），方向是把"每个回调的刷新"与"fd 对关联"解耦。尚未解决的边界是：如何按消费方 gate 哨兵，同时不悄悄破坏那些读取 pin 上下文 map 的外部 trace/profile 关联。
