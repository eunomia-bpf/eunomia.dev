# 为什么 GenAI 插桩应该用日志管道严重级来控制推理事件发射，而不是专门的 emit 标志？logger 的 enabled 是 opt-in 还是 opt-out？

门控应该放在标准日志管道里，而不是某个专门的环境变量标志上。GenAI 推理事件在规范中定为 DEBUG 严重级、Recommended 需求级别，因此保持普通 INFO 日志的管道本来就会丢掉它；插桩侧现在会在构造记录之前用 `Logger.enabled` 在该严重级上做一次判断，事件不通过时连分配和消息内容捕获都会跳过。关键的边界是 `Logger.enabled` 本身**不是** opt-in：在 SDK 默认日志器且没有配置最低严重级的情况下，它对一切严重级都返回 true。它只有在管道真正设了门槛之后才变成 opt-out 控制——最低严重级、trace 采样决策或事件名禁用——而且这个控制是声明式的，不是某个功能标志。

## 机制

OpenTelemetry 的 GenAI 规范把客户端推理事件定为 Recommended，并要求插桩在记录该事件时把严重级设为 DEBUG（severity number 5）。DEBUG 低于大多数日志管道使用的 INFO 门槛，因此这个事件默认就是"关着"的：普通部署不会收到它，除非你故意保留 DEBUG 级日志。规范明确该事件用于把输入/输出详情独立于 trace 保存下来，这正是为什么"构造它"的成本和触碰消息内容带来的隐私成本都很关键。

Python 的 GenAI 工具包正在把发射决策挪到这条管道上。它的 `_maybe_create_event` 路径现在会在该事件严重级上通过日志器判断事件是否启用，只有判断通过才会构造并发出日志记录，同时为那些 `enabled` 方法还不接受严重级参数的日志器留了优雅回退。专门的环境标志（`EMIT_EVENT` 一类）正在被移除，取而代之的是内容捕获标志加这个日志器判断：标志回答"要不要附带 prompt 和响应内容"，管道回答"这个事件会不会到达任何人"。

SDK 侧的支撑件是声明式配置：最低严重级和基于 trace 的过滤被接到了 `Logger.enabled` 上，声明式配置可以按插桩名禁用事件，并且还将支持按最低严重级或按 trace 采样标志禁用。把边界浓缩成一句话——功能标志回答"要不要碰内容"，管道回答"这个事件存不存得下来"，两者被分开，这样一个 DEBUG 级的诊断事件就不能悄悄跟着每条 INFO 日志一起被发出去。

## 验证与调试路径

1. 打开 GenAI client-inference 规范的 Events 部分：客户端推理事件是 Recommended 需求级，并且要求插桩在记录该事件时把严重级设为 DEBUG（severity number 5）。这一行就是该事件在 INFO 管道下默认静默的原因。
2. 读 opentelemetry-python 的 GenAI 工具包中解耦事件发射的改动：确认发射路径现在会按事件严重级去查 `logger.enabled`，返回 false 时跳过记录构造，并且插桩内部对环境的无条件标志写入已被移除。
3. 读 SDK 侧为 `Logger.enabled` 加入最低严重级和基于 trace 过滤、并接入声明式配置的改动：这正是把 `enabled` 从"默认返回 true"变成真正门控的那一步。
4. 在本地应用中，用带 INFO 门槛的日志管道确认 `enabled` 对该 DEBUG 事件返回 false、记录根本不会被构造；再调低门槛或在声明式配置中按事件名禁用它，观察行为切换。在内容捕获标志关闭时，确认即便事件被发出，其负载里也不含 prompt 和响应正文。

## 局限

- 没有配置最低严重级时，SDK 日志器的 `enabled` 对所有严重级都返回 true，所以 `Logger.enabled` 今天并不是独立的 opt-in；opt-in 的边界是"你的管道是否保留 DEBUG 事件"，而答案通常是"否"，除非你专门去配。
- 带严重级的 `enabled` 调用需要较新的 opentelemetry API 版本；不接受严重级参数的旧版或自定义日志器必须靠回退处理，否则判断会静默退化。
- 该事件目前是 Development 状态、并非所有语言都可用，同一模式在不同 SDK 上的支持并不均匀。
- 事件按设计携带输入/输出详情；内容捕获标志控制的是是否附带内容，而把事件本身挡在普通管道外的是严重级门控。

## 参考

- [OpenTelemetry semantic-conventions-genai — client inference events](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md)：客户端推理 operation-details 事件，Recommended 需求级，要求携带 DEBUG 严重级（number 5），可把请求与响应详情独立于 trace 保存。
- [opentelemetry-python pull request 5724](https://github.com/open-telemetry/opentelemetry-python/pull/5724)：为 `Logger.enabled` 加入最低严重级与基于 trace 的过滤，并接入声明式配置，同时支持按插桩名禁用事件。
- [opentelemetry-python-genai pull request 836](https://github.com/open-telemetry/opentelemetry-python-genai/pull/836)：GenAI 工具包解耦事件发射，把发射决策移到 DEBUG 严重级上的日志器判断，移除专门 emit 标志，改由内容捕获标志承担。
- [OpenTelemetry Python Logs API](https://opentelemetry.io/docs/languages/python/logs/)：Python 日志 API 表面，包括 `LoggerProvider`、`Logger` 与 `enabled`。

## 当日社区讨论

本 run 审查的两个 opt-in 存档渠道（两个 CNCF OpenTelemetry 插桩渠道）共产出十二条消息。最强的新话题就是本页回答的这个：Go 版 agent 开发工具链上的一个 pull request 被对照到 GenAI 工具包改动，被问及是否应沿用同一套日志器判断模式，悬而未决的问题是推理事件应保持 opt-in 还是改为 opt-out。帖内的答案就是上文那个边界：日志器判断默认是开着的，因此它本身不构成 opt-in 机制；opt-out 路径走的是声明式管道——按事件名、按最低严重级、或按 trace 采样标志禁用。第二个话题问是否计划为 Node.js 建一个与 Python 对应的 GenAI 插桩仓库；回复是插桩正落在 JavaScript contrib 仓库、自动生成规范落在核心 JS SDK，但尚无公开决定是否要专门仓库。另外两个话题延续此前已发布的内容：agent trace 的专门 skill-definitions 属性（10-05 条目）以及 Kubernetes model-serving-signals 计划（10-09 条目）。

本 run 的渠道覆盖：两个 opt-in 存档渠道共提供十二条消息。可见浏览器专属的渠道——eunomia-bpf 与 sched-ext 的 Discord 服务器、bpf 邮件列表、以及 r/eBPF——本 run 没有可用的可见浏览器会话，因此标记为"未覆盖"，而非"安静"。
