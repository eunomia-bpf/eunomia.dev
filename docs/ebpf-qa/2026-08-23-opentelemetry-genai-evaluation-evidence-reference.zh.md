# OpenTelemetry GenAI 评估结果是否应该携带可验证证据的引用？

**简短回答：** 用 digest 绑定的证据引用是 `gen_ai.evaluation.result` 一个有价值且范围合理的扩展，但它应当是可选字段，而且不能暗示 OpenTelemetry 已经验证了证据。当前 GenAI 约定记录 evaluation name、score 或 label、explanation、error class 和 response identifier；它**没有**定义 evidence URI、digest、media type、signature 或 verification result。

合适的实验设计应把证据保留在 telemetry 之外，只记录对精确字节的稳定引用：不含秘密的 URI、带算法的 digest 和 media type。Consumer 可以据此取回 evaluation receipt 并检查完整性，而不必把大型或敏感工件塞进每个 event。真实性、签名者身份、policy 和 trust 仍由被引用的证据格式及其 verifier 负责。

## 当前 event 回答“结果是什么”，而不是“证明在哪里”

当前 `gen_ai.evaluation.result` 是 recommended、development-status 的 event 约定。它要求 `gen_ai.evaluation.name`；在适用时记录 `gen_ai.evaluation.score.value`、`gen_ai.evaluation.score.label` 和 `error.type`；推荐记录 `gen_ai.evaluation.explanation`；当被评估 operation span 不可用时，用 `gen_ai.response.id` 关联 response。

这些字段足以展示“relevance: pass”一类结果，并把它关联到模型响应。但它们不能复现 evaluator、检查 signed receipt，或证明 evaluator 看过的是哪一组字节。Free-form explanation 是给人阅读的上下文，不是完整性锚点。

应当明确保持下面的分层：

```text
evaluation event   -> 结果和关联关系
evidence reference -> 外部精确字节的身份
evidence envelope  -> claims、signer、signature 和验证规则
policy decision    -> verifier 是否接受这些 claims
```

因此，增加 reference 是 additive change。它不会改变 score 的含义，也不应让轻量或在线 evaluation 被迫生成证据。

## 引用必须绑定哪些内容

最小的互操作形状包含三个逻辑字段：

| 字段 | 契约 |
| --- | --- |
| URI | 在不嵌入凭据的情况下标识 evidence object。规范必须说明它是 identity、retrieval location，还是两者兼具。 |
| Digest | 绑定精确字节，并包含算法，例如 `sha256:<hex>`。 |
| Media type | 告诉 consumer 如何解析取回的字节，但不声称内容有效或可信。 |

in-toto 的 `ResourceDescriptor` 是一个有用的公开先例：它区分 `uri`、`downloadLocation`、带算法的 digest map 和 `mediaType`。OpenTelemetry attributes 更扁平，因此初版提案可以使用一个在 value 中包含算法的 digest string。不要创建 `...digest.sha256` 这类 dynamic attribute key，也不要使用 index 可能错位的 URI、digest 和 media type 平行数组。

如果一个 result 有多个 supporting files，应让 event 指向列出这些文件及其 digest 的 manifest 或 signed envelope。这样 event 只保留一个原子引用，由 evidence format 自己承担 cardinality 和结构。

## 在标准确定前用 vendor namespace 试验

GenAI semantic convention 接受并发布正式名称以前，实现应使用自己的 vendor namespace。一个 prototype 可以表达下面的逻辑形状：

```text
event.name                                   = "gen_ai.evaluation.result"
gen_ai.evaluation.name                      = "policy_compliance"
gen_ai.evaluation.score.label               = "pass"
example.evaluation.evidence.uri             = "urn:example:evidence:01J..."
example.evaluation.evidence.digest          = "sha256:7f83b165..."
example.evaluation.evidence.media_type      = "application/json"
```

上面的 `example.*` 只是 placeholder，不是 OpenTelemetry attributes。生产 prototype 应说明：

- hash 的究竟是哪一种 byte representation，包括解压和 canonicalization 规则；
- 允许哪些 digest algorithm 和 encoding；
- URI 可以直接解析，还是必须经过内部 resolver；
- 一个 evaluation result 是否只能引用一个 evidence object；
- telemetry 与 evidence 之间的 retention 关系；
- object missing、expired、unauthorized 或 digest mismatch 对 consumer 分别意味着什么。

OpenTelemetry 的 convention-authoring guidance 建议新 attributes 在标准化前先跨实现 prototype。这个问题尤其需要至少一个在线 evaluator 和一个异步 evaluation pipeline 的证据，因为两者的关联和保留路径不同。

## In-flight 与 post-hoc evaluation 需要不同的 trace 关系

### In-flight evaluation

Evaluator 在 GenAI operation 活跃期间运行时，用该 operation 的 trace context 发出 `gen_ai.evaluation.result`。这符合当前“尽可能 parent 到被评估 operation”的建议。外部 evidence 可以在 event 前后紧邻写入，但在拿到已存储字节的最终 digest 前，不应导出包含引用的 event。

如果 evidence creation 失败，只有在 evaluation result 已独立确定时才记录结果。不要先发送 URI、稍后再补 digest：许多 telemetry pipeline 是 append-only，半填充 reference 会永久留下歧义。

### Post-hoc evaluation

不要重新打开或向已经 ended 的 span 追加数据。OpenTelemetry span 在结束后不能再记录新内容。应当：

1. 为稍后发生的 evaluation operation 创建 telemetry；
2. 在合法保留原始 `SpanContext` 时，让 evaluation span link 到它；
3. 可取得 response identifier 时设置 `gen_ai.response.id`；
4. 从后续 evaluation context 发出 result 和 evidence reference。

OpenTelemetry link 可以指向同一 trace 或另一 trace 的 `SpanContext`，比伪造 parent-child lifetime 更适合异步工作。Response identifier 是关联 fallback，不是 evidence integrity 的替代品。

## Digest 不是 attestation

Digest 只回答一个问题：“我取回的是不是 producer 引用的同一组字节？”它不能回答：

- 谁生成了这些字节；
- signer 是否得到授权；
- evaluator 是否真的执行了所声明的流程；
- hash 前的 inputs 是否完整、未被篡改；
- result 是否满足 consumer 的 policy。

如果需要这些保证，被引用对象应使用 DSSE/in-toto 等 authenticated envelope 或其他已记录的 attestation format。Verifier 必须检查 signature、signer identity、predicate type、subject、freshness 和本地 policy。除非 verification 本身被单独建模成一个结果，否则 telemetry event 仍然只能报告“存在一个 reference”。

这也避免一个危险命名：`verified=true` 会把 digest match、signature validity、trusted signer 和 policy acceptance 几个不同事实压成一个没有可移植含义的 boolean。

## 隐私和运维边界

Evidence reference 即使不含 prompt text，也具有高 cardinality，并可能是敏感数据。

- URI 中绝不能放 pre-signed URL、bearer token、user name、prompt fragment、tenant name 或私有 object-store topology。
- 优先使用 opaque identifier 或 content-addressed URI，并在授权环境内部解析。
- 把 URI 当作 opt-in telemetry，执行与 evidence 相同的 export、access 和 retention policy。
- 低熵或可猜内容的 hash 可能成为 confirmation oracle；digest 不等于匿名化。
- 不要把 evidence bytes 放进 span attributes 或 event body。大内容会增加 telemetry 成本，还可能绕过 evidence store 的 access control。
- 明确 trace 比 evidence 保留更久、或 evidence 比 trace 保留更久时的行为。Dangling reference 必须与 evaluation failed 区分开。

Media type 也只是 parsing hint。Consumer 应先按约定 byte representation 验证 digest，再解析内容，随后验证 payload schema，以及适用时的 authenticated envelope。

## 有界采用路径

1. **在 vendor namespace 中 prototype。** 每个 evaluation result 只用一个 reference，保持标准 `gen_ai.evaluation.*` 字段不变。
2. **覆盖两种时间模式。** 一个 evaluator 在 request 内运行，另一个在原 span 结束后运行。
3. **验证 byte identity。** 存储 evidence，从已存储 representation 计算 digest，通过 consumer 路径取回，再在解析前重算 digest。
4. **测试失败状态。** 覆盖 not found、unauthorized、expired、wrong media type、unsupported digest algorithm 和 digest mismatch。
5. **测量 telemetry 影响。** 确认 URI cardinality 不会被提升成 metrics，也不会被不加选择地索引。
6. **分开 integrity 与 trust。** 在 evidence verifier 中完成 signature 和 policy verification，不要把它们当成 telemetry reference 的固有属性。
7. **带互操作证据提交上游。** 在提出标准名称前，记录至少两个独立 producer 和 consumer、精确 privacy warning，以及单工件与多工件规则。

这个契约应保持克制：evaluation result 可以告诉 consumer 证据在哪里，以及应该看到哪些字节；它不应继续声称这些字节就是真相。

## 参考资料

- [OpenTelemetry GenAI event 约定，包括 `gen_ai.evaluation.result`](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md)
- [OpenTelemetry GenAI attribute registry](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md)
- [OpenTelemetry event 语义约定](https://opentelemetry.io/docs/specs/semconv/general/events/)
- [OpenTelemetry 定义和 prototype semantic conventions 的指南](https://opentelemetry.io/docs/specs/semconv/how-to-write-conventions/)
- [OpenTelemetry Trace API：span lifetime、events 和 links](https://opentelemetry.io/docs/specs/otel/trace/api/)
- [in-toto `ResourceDescriptor`：URI、digest、download location 与 media type](https://github.com/in-toto/attestation/blob/main/spec/v1/resource_descriptor.md)
- [in-toto envelope 规范与 authenticated payload 指南](https://github.com/in-toto/attestation/blob/main/spec/v1/envelope.md)
- [当前 OpenTelemetry GenAI agent 与 framework span 约定](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md)
- [OCI Image Manifest Specification 与 artifact guidance](https://github.com/opencontainers/image-spec/blob/main/manifest.md)
- [Linux BPF stream 实现](https://github.com/torvalds/linux/blob/master/kernel/bpf/stream.c)
- [Linux BPF token UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)
- [Linux BPF selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自过去 24 小时，因此没有使用七天 fallback。姓名、账号、雇主、workspace 和频道身份、message link、精确时间、私有拓扑、原始日志及可搜索回原讨论的措辞均已删除，也没有保留原始 transcript。

### Evaluation telemetry 需要完整性边界，但不应变成 evidence store

最强问题追问：GenAI evaluation result 是否应该指向 signed 或可离线检查的 receipt。上文给出了机制和安全形状：[当前 event convention](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md)可以报告 score 并关联 response，却没有 evidence-reference fields。URI、带算法的 digest 和 media type 可以补上缺口，前提是它们保持 opt-in，而且 digest 不被误写成 authenticity 或 policy approval。

下一步最有价值的工程验证是 two-producer prototype：一个 evaluator 在 model operation 活跃时发出结果，另一个异步运行。两者应生成相同的逻辑 reference；consumer 应取回 object、验证 digest、检查 media type，再把 signature 交给对应 verifier。尚未解决的边界是标准命名和 cardinality：对于多个 artifact，单一 manifest reference 比平行数组更容易查询和演进。

### Artifact packaging 和 trace rendering 都会因两套 schema 争夺 ownership 而失败

一个项目维护 feed 暴露了两个相关兼容问题。第一个涉及修改 OCI configuration descriptor，让 orchestration tooling 能识别 artifact，而已有 consumer 可能仍依赖旧 media type。[OCI manifest 规范](https://github.com/opencontainers/image-spec/blob/main/manifest.md)把 `config.mediaType`、`artifactType`、digest 和 content 当作一个整体契约；静默重写单个字段，会让不同 puller 对同一组 bytes 得到不同含义。安全诊断是检查实际 pushed manifest，并分别测试 registry storage、pull 和 runtime consumption。迁移应发布新的版本化 artifact shape，或使用标准 empty-config/artifact-type pattern，再为旧 consumer 保留有界兼容期。

第二个问题是：OTLP exporter 可以重建 model-call spans，却不一定生成 visualization 期望的外层 agent 或 workflow spans。因此，“export 成功”不等于“trace hierarchy 完整”。应把实际 span name、kind、parent 和 `gen_ai.operation.name` 与[当前 agent-span 约定](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md)逐项比较，再区分“model calls 可见”和“agent lifecycle 已建模”。未决工作是确定哪一层能够可靠观察 agent boundary；如果 exporter 只能看到 HTTP traffic，就不应凭空构造无法证明语义的 `invoke_agent` root。

### Kernel review 集中在 failure-path accounting 与 delegated capability limit

当日公开 kernel 工作反复检查 success path 之外的不变量。BPF stream changes 涉及 allocation failure 后的 capacity reservation、staged-write accounting、userspace read fault 后的 partial progress，以及超过 backing buffer 的 output rejection。[当前 stream implementation](https://github.com/torvalds/linux/blob/master/kernel/bpf/stream.c)是主要代码边界。实际测试规则应当是 transactional：每次 allocation 或 copy 失败后，reported capacity 必须等于 allocated storage，reserved bytes 只能归还一次，已经复制出一部分数据的 read 不应抹掉进度。Failure injection 与 oversize case 应进入 [BPF selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf)，不能只停留在 review 推理中。

另一项 review 讨论 delegated BPF-token program 是否应该访问 connection-tracking kfuncs；这些 kfunc 的 capability check 可能假设更高权限的 network context。[BPF token UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)在 user namespace 内委托选定 commands、map types、program types 和 attach types，但这不会自动让每个 kfunc 都适合 token-loaded program。下一步诊断是针对每个暴露的 program-type/kfunc 组合，同时测试 token 与 non-token loading，并确认 capability check 发生在预期 namespace。未决边界是 per-kfunc delegation policy；review-stage restriction 在合入前不能被写成 released kernel guarantee。

### 安静目标仍然完成了检查

Scheduler help surfaces、公开 practitioner forum 和多数 project-specific support channels 在当日窗口内没有新的实质工程交流。一个 networking surface 只有社区日程安排，若干 project channels 只有介绍或自动维护通知。它们全部可访问，被记录为安静而不是跳过。近期一篇关于 crash-safe BPF loader reconciliation 的公开帖子不在 24 小时窗口内，因此无需作为 fallback 使用。
