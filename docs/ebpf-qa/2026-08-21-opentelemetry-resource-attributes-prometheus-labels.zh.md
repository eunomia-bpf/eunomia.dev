# OpenTelemetry 指标生产者是否应该把服务身份复制到每个数据点？

**简短回答：** 在规范的 OTLP 输出中不应该。`service.name`、`service.namespace` 和 `service.instance.id` 描述的是产生 telemetry 的实体，因此应放在 OpenTelemetry `Resource` 上。把它们重复写入每个 metric point，会制造第二个事实来源，并把资源到后端的映射问题变成 producer 私有的 telemetry 形状。

Prometheus 是需要在**导出边界**处理的重要例外，而不是在 producer 边界处理。Prometheus 没有原生的 OpenTelemetry Resource 对象，因此 exporter 可以把服务身份转换为惯用的 `job` 和 `instance` target label，把其他选定的资源属性放进 `target_info`，或有意识地把一小份 allowlist 提升为 metric label。这种转换应当显式且只服务于特定后端。即使同一个进程还暴露 Prometheus endpoint，OTLP producer 也应保持规范形状。

## Resource 属性和 point 属性回答不同问题

OTLP metrics 不是一组扁平的带 label sample。协议把 `ScopeMetrics` 放在 `ResourceMetrics` 下，每个 `ResourceMetrics` 消息携带一个 Resource。OpenTelemetry metrics data model 也把来源 Resource 作为 metric stream identity 的一部分，另外还包括 instrumentation scope、metric name 和数据点自己的 attributes。

两者语义不同：

- **Resource attributes** 回答“哪个实体产生了这份 telemetry？”典型例子包括服务、主机、云区域、容器和部署环境身份。
- **Data-point attributes** 回答“这个测量值描述的是哪个维度？”典型例子包括 HTTP method、status code、RPC method、device 或 queue。

服务语义约定把 `service.name`、`service.namespace` 和 `service.instance.id` 定义为服务实例的身份三元组。把三者搬到或复制到每个数据点，并不会给 OTLP 增加信息，反而改变 metric stream，增加重复 payload 和 label 处理，还允许 Resource 与 point 的值互相矛盾。

可以用下面的不变量约束设计：

```text
一个被观测的服务身份 -> 一个规范 Resource
测量维度             -> data-point attributes
后端兼容             -> exporter translation
```

如果数据点已经带有与 Resource 同名的 key，应在部署前定义优先级。让后端碰巧保留的值获胜，会使查询语义依赖 exporter 执行顺序，而不是 telemetry 模型。

## 为什么 Prometheus 需要转换

Prometheus 存储带 label 的 time series 和 scrape target 元数据；它不会在每组 metrics 旁携带 OTLP Resource。因此，OpenTelemetry 的 Prometheus/OpenMetrics 兼容性规范定义了映射，而不是要求 producer 自行扁平化 Resource。

对服务身份，惯用映射是：

```text
service.namespace + service.name -> job
service.instance.id               -> instance
其他 resource attributes          -> target_info 或选定 labels
```

`target_info` 是用 labels 描述 target 的 info metric。每个 `(job, instance)` 有一条一致的 series 时，PromQL 可以按需把资源元数据 join 到业务 metrics 上。这样可以保留查询能力，同时不把资源元数据复制进每条存储 series。

Collector 的 Prometheus exporter 还提供 `resource_to_telemetry_conversion.enabled`。开启后，它会把**所有** Resource attributes 复制为 metric labels；文档中的默认值是 `false`。这是兼容开关，不是普遍安全的默认方案。全量提升会放大 series 数量，暴露原本不应成为 metric 维度的属性，与 point 属性发生冲突，并让 dashboard 无意中依赖某个后端特有的扁平化行为。Exporter 文档建议在确实需要时用 transform 选择性复制常用属性。

## 让两条输出路径彼此独立

同时支持 OTLP push 和 Prometheus scrape endpoint 的 producer，应把它们看成对同一个内部 Resource 的两种 encoder，而不是让两者复用一份扁平 metric 表示。

### OTLP 路径

1. 把服务身份和其他实体元数据放在 Resource 上。
2. 数据点只保留测量维度。
3. 向 Collector 或后端发送规范的 `ResourceMetrics`。
4. 由接收 pipeline 根据自己的存储模型决定是否提升资源属性。

### Prometheus 路径

1. 从同一份内部服务身份派生 `job` 和 `instance`。
2. 在选定兼容 profile 要求时，生成一条一致的 `target_info` series。
3. 只提升明确 allowlist 中的其他 Resource attributes。
4. 确定性处理 label name 冲突，并记录规则。

即使两个 exporter 都在同一 binary 中，这种分离仍然重要。一个关于 eBPF OpenTelemetry instrumentation 的当前上游设计 issue 记录了 OTLP 与 Prometheus exporter 独立演进后可能出现的偏差：名称映射不同、target identity 丢失，以及 producer 自己生成的 info metric 与 exporter 生成的 `target_info` 冲突。该 issue 仍是 open 状态，所以它证明的是兼容性问题和提议方向，不能被当作已经交付的行为。

## 不打断 dashboard 的迁移方案

删除重复服务 labels 前，应先记录每条路径的真实契约。

1. **按结构检查 OTLP。** 确认服务身份只出现在 Resource 上，point attributes 只包含预期测量维度。
2. **检查 Prometheus endpoint。** 记录代表性 counter、histogram 和 `target_info` 的 labels，确认只有一套一致的 target identity。
3. **盘点查询和 recording rules。** 区分直接按复制服务 label 分组的查询，以及使用 `job`、`instance` 或 info-metric join 的查询。
4. **选定稳定 Prometheus profile。** 定义 `job`/`instance` 映射、Resource attribute 的精确 allowlist、冲突优先级，以及是否生成 `target_info`。
5. **Canary exporter 变化。** 比较 series 数、label cardinality、scrape 大小、missing-series alert 和代表性 dashboard 结果。
6. **最后删除 producer 重复字段。** 如果 consumer 无法原子迁移，可以保留有界兼容期，然后删除旧 point labels，不要永久维护两套身份。

测试不仅要断言存在，也要断言缺失。如果服务身份 key 意外重新出现在每个 OTLP point 上、两个不同 Resource 被折叠为同一个 Prometheus `(job, instance)`，或两个 `target_info` producer 产生冲突 series，conformance fixture 都应失败。

## 修复应该放在哪一层？

可以使用下面的 ownership 规则：

- 输出是 OTLP 时，修复 producer，使其发出正确 Resource，并停止把资源身份复制进 points。
- Prometheus consumer 需要把资源字段当 label 时，配置或修复 Prometheus exporter。
- 不同后端需要不同投影时，在各自的 Collector pipeline 中转换。
- Dashboard 依赖意外重复字段时，迁移 dashboard，不要围绕这个偶然行为重定义规范 telemetry 模型。

目标不是禁止 label，而是维持一套权威服务身份，并让每次扁平化都可见、可测试、可回滚。

## 参考资料

- [OpenTelemetry 服务 Resource 语义约定](https://opentelemetry.io/docs/specs/semconv/resource/service/)
- [OTLP metrics 协议：`ResourceMetrics` 把 Resource 与 scope metrics 分组](https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/metrics/v1/metrics.proto)
- [OpenTelemetry Metrics Data Model](https://opentelemetry.io/docs/specs/otel/metrics/data-model/)
- [OpenTelemetry Prometheus 与 OpenMetrics 兼容性规范](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/compatibility/prometheus_and_openmetrics.md)
- [OpenTelemetry Collector Prometheus exporter 配置](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/exporter/prometheusexporter/README.md)
- [关于统一 OTLP 与 Prometheus metric resources 的 eBPF instrumentation open issue](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2974)
- [当前 OpenTelemetry GenAI span 约定](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [关于 cost attributes 的 GenAI 语义约定 open issue](https://github.com/open-telemetry/semantic-conventions-genai/issues/101)
- [BPF 线程：更新 cgroup BPF link 时验证 attach type](https://lore.kernel.org/bpf/1ab678ef-6349-4374-9ebf-22f857211ca7@linux.dev/T/#t)
- [BPF 线程：让 arena page fault 在 memory cgroup 限制下可以 reclaim](https://lore.kernel.org/bpf/20260821050250.35112-1-jiayuan.chen@linux.dev/T/#t)
- [BPF 线程：保持 AF_XDP transmit-metadata ABI 与 batching state 正确](https://lore.kernel.org/bpf/20260819160535.1472459-1-sdf@fomichev.me/T/#t)
- [RFC 1982：serial-number arithmetic](https://www.rfc-editor.org/rfc/rfc1982.html)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自过去 24 小时，因此没有使用七天 fallback。姓名、账号、雇主、workspace 和频道身份、封闭聊天链接、精确时间、私有拓扑、原始日志及可搜索回原讨论的措辞均已删除，也没有保留原始 transcript。

### Metric producer 和后端需要明确谁拥有服务身份

最主要的讨论涉及一个基于 eBPF 的 metrics producer，它同时把相同服务字段表示为 OTLP Resource 和 point attributes。讨论者希望保留 Prometheus 中有用的 labels，同时避免让 OTLP consumer 为重复元数据付出代价。机制就是上文的模型差异：OTLP 有 `ResourceMetrics`，Prometheus 则需要 target labels 或 info metric。

直接诊断方法是同时比较 OTLP envelope、scrape 输出和 Collector 的 resource-conversion 设置。兼容路径应保留规范 Resource attributes，在 Prometheus 侧把服务身份映射为 `job` 和 `instance`，并只提升已经证明需要的查询维度。尚未解决的设计问题是：多少映射应该由内置 exporter 配置，多少交给 Collector；[上游对齐 issue](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2974)仍为 open，不能表述为已实现功能。

### GenAI cost telemetry 仍缺少已发布的共同形状

另一条可观测性问题追问：模型调用成本是否应该有标准 span attribute。一些 instrumentation 已经发送 vendor-specific cost fields，但[当前 GenAI span convention](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)还没有定义稳定的通用成本属性。一项[仍然 open 的语义约定 issue](https://github.com/open-telemetry/semantic-conventions-genai/issues/101)还在讨论成本应属于核心约定还是 extension，以及单位应如何表示。

目前应让 vendor 字段保持 namespace，派生成本时记录 currency 和 pricing-table version，并只在版本化内部 schema 中规范化。约定尚未解决单位和 ownership 前，不要把 vendor 字段改名为标准 OpenTelemetry 属性。未决问题不只是 attribute name：跨 provider 比较还需要币种、生效价格日期，以及 cached 或 discounted tokens 的计价规则。

### 内核 review 聚焦 replacement 和 fault 边界的兼容性检查

公开 BPF 工作展示了三种“fast path 成功，但 replacement 或 pressure path 仍不安全”的情况。Cgroup BPF-link update 检查了宽泛的 program type，却可能接受不兼容的 attach flavor；提议修复会在 replacement 时验证 attach type。Arena page fault 遇到 memory-cgroup limit 后可能进入不可 reclaim 的 allocation path，把有效但受压力影响的 fault 报为 `SIGSEGV`；提议方向是在锁外完成可 reclaim allocation。AF_XDP transmit-metadata review 则发现跨 ABI padding 问题，以及 zero-copy batching 复用 descriptor 时必须保持正确的状态。

公开证据包括 [cgroup-link update 讨论](https://lore.kernel.org/bpf/1ab678ef-6349-4374-9ebf-22f857211ca7@linux.dev/T/#t)、[arena memory-pressure 系列](https://lore.kernel.org/bpf/20260821050250.35112-1-jiayuan.chen@linux.dev/T/#t)和 [AF_XDP metadata 系列](https://lore.kernel.org/bpf/20260819160535.1472459-1-sdf@fomichev.me/T/#t)。这些是 review 阶段的 upstream changes，不代表每个 released kernel。实用测试矩阵应覆盖同类型但不兼容 attach flavor 的 replacement、`memory.max` 下的有效 fault、32/64-bit metadata layout、metadata-enabled flag，以及 metadata state 不同的连续 batched packets。

### 可靠 telemetry stream 必须把顺序绑定到 producer identity

一个项目维护 feed 中的 review 问题涉及 gap marker、replay、resume token、capability advertisement 和 frame ceiling。它们共享一条协议规则：control information 必须和它所描述的数据处于同一个排序与 replay domain，而 sequence number 只有在已知 producer incarnation 内才有意义。[RFC 1982](https://www.rfc-editor.org/rfc/rfc1982.html)给出了有界 serial-number arithmetic，但应用协议仍需定义 restart identity、replay retention 和 negotiation limits。

实用检查包括：旧消息仍在 buffer 时注入 queue gap、producer restart 后 resume、某条 capture path 不可用时启动，以及发送恰好达到协商上限的 frame。Review findings 不等于 released-product behavior；未决工作是先把这些不变量变成协议测试，再宣传连续性保证。

其余调度、项目支持、网络和公开论坛目标在当日窗口内没有新的实质技术交流，或只有例行介绍和自动通知。它们全部可访问，并被记录为安静，而不是跳过。
