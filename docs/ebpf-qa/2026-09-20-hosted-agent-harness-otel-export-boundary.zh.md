# 当智能体循环跑在厂商托管服务里而不是你的进程内时，如何从托管智能体 harness 里取出可观测数据？

**简短回答：** 你没法用 eBPF 插桩它，因为没有本地进程可挂——智能体循环、模型调用、工具执行都发生在厂商的托管服务里，而不是你的机器上。你能从自己这一侧观测到的只有 HTTP 边界：通用客户端插桩（eBPF 或用户态客户端 tracer）仍然能看到每个出站 `POST /v1/agents/runs` 请求以及它返回的流式响应，所以请求/响应大小、延迟、状态都保留下来了。要重建*内部* span 树（turn、模型响应、工具调用），你得依赖厂商自己的导出路径：OpenAI 的 Agents API 把每个 session 的步骤记录成 OpenTelemetry 的 **span**（"the trace groups model responses and tool calls under the root agent or the subagent that performed them. Each recorded step is called a **span**"），并允许你从公开端点*拉取*这些 trace 为 OTLP JSON。这就是承重边界：trace 不是你自己发射进流水线的东西，而是你按需求从厂商拉取的东西。

## 为什么 eBPF 够不到内部循环

eBPF 挂到你*宿主机*上某进程的 kernel 执行。托管智能体 harness 正好相反：你的代码向厂商服务发一个 HTTP 请求，之后所有有内容的事——规划、每个模型响应、每次工具调用、子智能体委派——都在厂商那一侧执行。没有本地 kernel trace。这正是它区别于其他反复出现主题的类别：Claude / OpenAI *SDK* 的情形可插桩，因为那个 SDK 的智能体循环跑在你机器上，它的 eBPF 或客户端插桩能在本地捕获内部 `chat` / `execute_tool` 交叉；而*托管服务*情形不行，因为循环在一个你只能跨越、无法插桩的 HTTP 边界背后。边界就是进程局部性，它决定了你在厂商不为你导出结构时能重建什么、不能重建什么。

## 跨过边界的三样东西

只有两条数据路径跨过 HTTP 边界，两者都对普通客户端追踪可见：

1. **出站请求。** 智能体运行是你的进程发出的一个请求；eBPF（或用户态客户端 tracer）能看到它的大小、耗时、状态。这就是*本地*信号的全部。
2. **实时事件流。** 每个 session 都暴露一个事件流，实时展示智能体进展。对 liveness 或轻量 UI 有用，但它不是结构化 span 集合，不能查询、关联或入库；而且智能体的回答可能先于它的 trace 与 token 用量就绪就出现。
3. **录好的 trace。** 事后，厂商存下该 session 的 span。这是唯一承载*内部*结构的路径，且是拉取式的：`GET /v1/agents/sessions/{session_id}/traces` 返回一页 OTLP JSON 的 trace。指南明确写道"Trace export must be enabled for your organization"，且 key 需要 `api.traces.read` 或更宽的 `api.agents.read` 权限。

决定性不对称在于：内部树只在厂商选择导出的范围内存在。托管 harness 是个黑盒，除非它把 span 结构交给你；而交付方式是*拉取*一个厂商托管的产物，受厂商设定的权限与组织闸门控制，而不是你配置、指向你 collector 的 exporter。

## 如何核实你能不能重建什么

1. **假设丢东西之前先确认循环是远端的。** 判定智能体是跑在你的进程里（SDK）还是托管服务里（托管 API）。若是托管服务，内部 span 树按构造就不在你的宿主机上。
2. **插桩交叉点，而非内部。** 在出站请求上加客户端追踪；这一层只能得到任何 eBPF 或用户态客户端 tracer 都会给的普通 HTTP 信号（延迟、状态、载荷大小）。别指望模型/工具 span 树出现在这一层。
3. **在厂商提供处拉取 trace。** 对 OpenAI 的 Agents API，用组织已启用导出、且 key 有读权限的方式，从 session trace 端点取回 OTLP JSON，再导入任何 OpenTelemetry 兼容的后端。把它当*恢复来的* trace，而非*原生发出的*：它的 span 词汇是厂商的，它的时序是厂商的，且只为厂商记录过的 session 出现。
4. **留意"没有 exporter"的情形。** 某厂商 beta 若不暴露 tracing 配置与外部 trace exporter，那你实时的只有事件流、事后的只有录好的 trace——两者都在厂商手里。对其他云托管智能体服务，泛化后成立：一旦循环离开你的进程边界，可观测的就只有 HTTP 交叉与厂商选择导出的部分。

## 决定它的边界

答案由**局部性**界定。eBPF（以及任何用户态客户端插桩）只能在离开你进程的交叉点上重建 trace 图；它无法制造出在远端服务里运行的循环的内部。内部 span 结构*只有*在 harness 暴露 exporter 或 trace 导出路径时才可得——而那个暴露是厂商的决定，受其权限约束、按组织启用。所以实用规则是：对托管 harness，按 (a) 出站 HTTP 交叉作为可靠本地信号，(b) 厂商事件流作为实时进展，(c) 厂商录好的 trace 导出作为完成 span 树来规划；把"没有外部 exporter"当作一条硬边界，而非可本地填上的缺口。

## 参考资料

- [OpenAI Agents API — Observability（每个 session 暴露事件流；session 日志与 token 用量可在 dashboard 查看；录好的 trace 可在那里检查或导出为 OTLP JSON）](https://developers.openai.com/api/docs/guides/agents-api/observability)
- [OpenAI Agents API — Tracing（tracing 默认启用；每个录好的步骤是一个 span；`GET /v1/agents/sessions/{session_id}/traces` 返回 OTLP JSON；导出须为组织启用，且 key 需有 traces-read 权限）](https://developers.openai.com/api/docs/guides/agents-api/tracing)
- [OpenAI Agents API — Export session traces（OTLP JSON 导出端点详情）](https://developers.openai.com/api/docs/guides/agents-api/tracing#export-session-traces)
- [OpenAI Agents API — Live session events（进行中的事件流）](https://developers.openai.com/api/docs/guides/agents-api/sessions/events)
- [W3C — Context Propagation（`traceparent`/`tracestate` 契约，让 trace 可跨进程与厂商边界携带）](https://www.w3.org/TR/context-propagation/)
- [OpenTelemetry — Traces 概念（span、一个 trace 导出将承载的信号）](https://opentelemetry.io/docs/concepts/signals/traces/)

## 当日社区讨论

本监控窗口是跨两个已 opt-in 只读归档的 Slack 归档（均为 OpenTelemetry 插桩频道）的滚动一周；两个可见浏览器聊天工作区与公共邮件列表/子版表面本次未审阅，该缺口已如实记录而不视为安静。若干主题自前几日的窗口延续而来；其中有新边界、且有源码确证的，是上述托管智能体可观测性问题。

**托管智能体 harness 与可观测性边界（即上文问题）。** 一位使用者在看到托管 Agents API 发布后询问如何取出可观测数据，并担心这种思路若流行起来，会像云厂商对待托管服务那样把数据锁住。可行答案是局部性边界：eBPF 插桩是本进程的，跑在厂商服务里的循环无法像在 *SDK* 情形那样插桩；只有离开进程的 HTTP 交叉可见。决定性、有源码依据的细节是厂商自己的导出路径——OpenAI 把每个步骤录成 OpenTelemetry span，并提供按组织、按权限闸门的拉取式 OTLP JSON 导出——这是"从厂商拉 trace"的模型，不是"配置外部 exporter 进你的 collector"的模型。未解决的边界在于某厂商 beta 若不暴露 tracing 配置与外部 exporter：只有事件流与录好的 trace，且都在厂商手里。

**GenAI 智能体 span 形状：兄弟 vs 嵌套（延续）。** 一位维护者确认：一个 turn 一个 `invoke_agent` span，`chat` 与 `execute_tool` 作为其直接子节点、以工具调用 id 关联，是预期的形状；把每次工具调用嵌套在请求它的 `chat` 下，是语义规范里因果树的想法，只有能把上下文经 sidecar 携带时才有用，目前并非必需。兄弟加调用 id 的形状不算错。

**OBI 配置 v1→v2 迁移与 Helm chart 缺口（延续）。** 一位使用者仍撞上 `fields are outside the supported v1-to-v2 migration contract`，想要中间地带；维护者立场不变：全有或无的 `migrate` 是故意的，逃生舱是显式的部分/尽力模式——能迁的迁好、剩下的报告出来，而非静默默认。相关的 OBI Helm-chart v2 缺口（chart 的 helpers 仍向 v2 配置 schema 注入一个 v1 字段）被要求在 helm-charts 仓库单独开 issue，附上 chart/OBI 版本、相关 values、渲染配置与校验错误。

**Node.js 哨兵成本与智能体生命周期（延续）。** Node.js 成本线继续：eBPF 探针很便宜；成本在注入的 Node 智能体每次回调的哨兵。维护者确认了该读法：客户端 span 父化来自 fd 对 map，每次回调的哨兵让 trace 上下文 map 与活动请求对齐，第三个消费者（外部 trace/profile 关联）也读这个固定的 map——所以只按手动 span 与日志富化来 gate 哨兵，可能悄悄破坏那个集成。相关生命周期报告：一位使用者把某服务排除出追踪后，须做 pod 重启才看到延迟下降，因为注入的智能体一直跑到 pod 被回收。
