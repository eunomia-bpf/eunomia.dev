# OpenInference 应如何与 OpenTelemetry 的 GenAI 语义约定共存？

**简短回答：**把 OpenTelemetry 作为传输与生命周期基础，把 OpenTelemetry GenAI 语义约定作为互操作目标，并把 OpenInference 视为已捐赠的成熟插桩库与兼容来源，后续由社区逐项整合。代码捐赠并不会让所有现有 OpenInference span 在一夜之间与 OpenTelemetry GenAI span 语义完全相同。

OpenInference 已经生成正常的 OpenTelemetry 数据并可通过 OTLP 导出，优势是覆盖大量 AI 框架和成熟的专用插桩。OpenTelemetry GenAI 项目定义模型调用、agent、MCP、指标和事件的厂商中立名称与结构。两者可以互补，但属性名和建模选择可能不同。迁移期间应在 collector 边界选择一种规范化 schema，并测试最终遥测；不要对同一个 SDK 同时启用两套插桩。

## 捐赠究竟改变了什么

OpenTelemetry 官方捐赠仓库说明已经接收 OpenInference 代码授权。其流程明确是增量式的：通过一次性贡献接收原始代码，归档接收仓库，再由 GenAI SIG 逐个 cherry-pick 插桩。

这建立了治理关系和代码复用路径，但不表示：

- 所有软件包已经迁入最终 OpenTelemetry 命名空间；
- 所有 OpenInference 属性已经匹配当前 OTel GenAI schema；
- 用户应对同一 SDK 安装两套插桩；
- 已存储的历史 trace 会被自动重写。

OpenTelemetry GenAI 语义约定现已位于独立官方仓库，扩展核心语义约定并覆盖 span、指标、事件、agent、MCP 和厂商专用约定。新建可互操作遥测应以它作为公开契约。OpenInference 仍是实际维护的插桩生态，其仓库也明确支持任何兼容 OTLP 的 collector。

## 区分线协议兼容与语义兼容

OTLP 回答“collector 能否收到这些遥测”；语义约定回答“不同生产者是否用同一字段表达同一含义”。两个库都能生成有效 OTLP，却可能用不同 key、span 名称或值编码表达同一概念。

例如，兼容层可能要把 OpenInference 的 span-kind 属性转换成 OTel 的 `gen_ai.operation.name`，而不是只改包名。已获接纳的 collector 组件讨论把它定义为传输途中的属性规范化，因为数据源已经使用标准 OTLP receiver。合理边界是：

```text
应用插桩
   ↓ OTLP
collector 规范化与脱敏
   ↓ 统一 OTel GenAI schema
后端、仪表盘与告警
```

规范化必须显式并带版本。映射应记录源 profile、目标语义约定版本、是否保留原字段，以及 value 如何转换。盲目改 key 可能破坏含义，尤其当一个源值对应多个 OTel operation，或事件 payload 带有不同隐私风险时。

## 安全迁移方案

首先盘点每个 SDK 实际由哪个库插桩，并确认应用、框架或厂商是否已经原生生成遥测。重复插桩通常比临时 schema 不一致更糟，会产生嵌套或重复 span、重复 token 计数和模糊的错误归因。

然后选定目标版本并建立小型一致性语料：

1. 一次成功模型调用；
2. 一次流式响应；
3. 一次工具调用与结果；
4. 一次 agent handoff 或嵌套调用；
5. 一次失败；
6. 一次禁用正文采集或仅保存外部引用的调用。

逐条验证 operation name、厂商和模型标识、请求/响应 token、finish reason、错误状态、工具调用关联和会话内容策略。比较 trace 语义，而不是序列化 JSON 的字段顺序。

过渡期优先选择三种模式之一：

- 保留 OpenInference 插桩，在 collector 中规范化；
- 一致性测试通过后替换为已被 OpenTelemetry 接纳的插桩；
- 在限定兼容期内同时保留源字段和规范字段。

不要为了生产对比而让两套生产者同时插桩同一 SDK。确需影子比较时，应放在测试 workload 中，或禁止其中一条 pipeline 导出。

## 隐私与发布限制

GenAI 遥测可能包含 prompt、response、工具参数、检索文档和标识符。schema 统一不代表这些字段安全。应在插桩端控制内容采集，在 collector 强制脱敏和大小限制，并单独配置后端留存。当应用已安全保存内容时，优先记录引用而不是正文。

剩余不确定性是各软件包的迁移时间。官方流程明确允许逐项迁移，因此用户必须检查目标 SDK 当前的软件包仓库和 release note。“OpenInference 已捐赠”本身不足以作为升级指令。

## 参考资料

- [OpenTelemetry：OpenInference 捐赠仓库与整合流程](https://github.com/open-telemetry/donation-openinference)
- [OpenTelemetry GenAI 语义约定仓库](https://github.com/open-telemetry/semantic-conventions-genai)
- [OpenTelemetry 文档：GenAI 约定已迁入独立仓库](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OpenInference 仓库：OpenTelemetry 插桩与 OTLP 目标](https://github.com/Arize-ai/openinference)
- [OpenTelemetry Collector 提案：规范化通过 OTLP 发送的 GenAI 属性](https://github.com/open-telemetry/opentelemetry-collector-contrib/issues/46069)
- [BPF 邮件列表：重新设计 verifier 诊断](https://lore.kernel.org/bpf/a4e7eebf34507bf3041f232561e6f0a8acd47d7f.camel@gmail.com/T/#t)
- [BPF 邮件列表：loader 文件描述符提案](https://lore.kernel.org/bpf/20260813002618.3755631-1-tweek@google.com/T/#t)

## 今日社区讨论

今天通过普通可见界面检查了全部 6 个获准社区、共 15 个 allowlist 频道或公开页面；所有目标均可访问。入选问题来自过去 24 小时，因此没有使用 7 天回退。以下内容已删除姓名、账号、雇主、频道身份、消息链接、精确时间、私有拓扑、原始日志与可回搜措辞；没有保留原始 transcript。

### GenAI 可观测性已经成为迁移问题，而不只是覆盖问题

最强讨论询问 OpenTelemetry 接收 OpenInference 后是否仍在构建一个竞争性的 GenAI 框架。混淆来自把三层概念混为一谈：OpenTelemetry 遥测 API 与 OTLP 传输、GenAI 语义契约，以及面向具体 SDK 的插桩库。捐赠主要改变所有权并提供成熟插桩代码；独立 OTel 仓库仍是共享 GenAI 语义的规范来源。

同日相关讨论暴露了实际后果：旧 Python 插桩正在弃用，而上游 SDK 发布了新的主版本。问题是弃用包在最后一个版本中是否应临时限制 SDK 版本。软件包所有权、schema 所有权和兼容策略必须分开：弃用通知不会防止破坏性依赖升级，语义映射也无法修复不兼容的 monkey patch。维护者需要有界版本限制、明确后继包，以及覆盖最后支持版本和第一个不支持版本的测试。

实践中的排障路径是：确认进程实际加载的插桩包，记录其版本和目标 SDK 版本，采集一条最小 trace，同时检查 span 重复和属性形态。如果使用 collector normalizer，应独立测试它的输出。尚不确定的是每个捐赠库的接纳时间；官方流程有意让它们逐项迁移。

### 运行时 eBPF 插桩安静，项目自动化活跃

eBPF 插桩社区在当日窗口内没有新的实质技术消息。近期可见内容仍围绕运行时 eBPF 插桩与编译期语言插桩的选择。边界仍是可附加性与源码级保真度：eBPF 可以观察未修改进程和大规模节点，而编译期插桩能提供内核可见信号无法可靠重建的语言和框架上下文。

多个项目专项频道较安静。一个开发通知频道主要出现自动测试、构建和预览部署，而不是用户问题。项目通用区域没有新技术请求，调度器支持区在此前命令行参数事件后也没有当日活动。这些页面都已检查并计为安静，没有被用来虚构话题。

### 上游 BPF 聚焦诊断、所有权和失败路径

公开 BPF 归档非常活跃。一组 verifier 诊断补丁为寄存器类型安全、内存边界、资源生命周期、调用参数、执行上下文、控制流结构、策略和 verifier 限制提供结构化类别与源码/指令上下文。它直接回应长期体验问题：拒绝通常正确，但日志没有清楚区分根因和后续传播错误。实践上应保留第一个诊断事件，将它关联到源码和指令位置，而不是只把最终寄存器状态当作解释。

其他线程涉及从已打开文件描述符加载 BPF 对象、终止 `LDIMM64` 重定位可能越界、多次 detach 失败后的 trampoline image 生命周期、栈深统计、可 mmap array map 的惰性填充，以及 16 字节聚合返回值。共同机制是部分成功或失败时的边界所有权：文件、重定位指令对、trampoline image、verification root 或返回寄存器对由谁持有。测试必须覆盖取消与回滚，而不只是成功路径。

公开论坛最新可见内容介绍了一个 XDP/TC DDoS 项目，而不是新的排障问题。其架构区分 host 与 router 模式，并提供 dry-run，这是有用的部署模式：先观察计数器和决策，再启用 drop。通用 eBPF 讨论区没有新的当日技术问题，最近的 socket-map 主题也已在先前 Q&A 中回答。综合来看，今天从 agent 遥测到内核 BPF 的共同系统教训是：互操作需要明确所有权、版本化契约和失败路径测试，而不只是共享传输协议或一次成功 load。
