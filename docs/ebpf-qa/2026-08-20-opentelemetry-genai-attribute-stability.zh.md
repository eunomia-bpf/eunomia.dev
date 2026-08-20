# 如何判断 OpenTelemetry GenAI 属性是否已经稳定到可以依赖？

**简短回答：**不要因为某个属性出现在生成的 registry 中、某个 instrumentation 包已经发出它，或相关 PR 已获得认可，就推断它已经稳定。应先检查描述该语义约定的具体文档状态，再检查定义是否进入正式发布，以及实际生产 telemetry 的 instrumentation 是否承诺输出形状稳定。

截至本文写作时，OpenTelemetry 官方 GenAI 语义约定总览仍标为 **Development**。独立仓库还没有正式 release，changelog 只有 `Unreleased`，顶层 schema URL 也尚未给出。因此，当前 GenAI 名称和 signal 结构适合实验和边界清晰的生产试用，但还不是通用的 GA 兼容性契约。应固定所消费的 profile，在转换层后使用，并预留迁移工作。

## 四种容易混淆的状态信号

OpenTelemetry 中有几种彼此独立的“稳定性”，它们回答的问题不同。

### 1. 文档状态决定它所描述约定的成熟度

OpenTelemetry 的文档状态规则明确说明：状态只作用于具体文档，不自动扩展到整份规范。`Development` 表示组件尚未完整，可能频繁变化，不应被当作生产就绪能力，甚至可能在不预先通知的情况下移除；`Release Candidate` 会严格限制破坏性变化；`Stable` 才是通用可用级别。

GenAI 总览目前写的是 `Status: Development`。生成的 registry 列出了许多 `gen_ai.*` key，但 Stability 单元格为空。表格中出现一个 key，只能证明当前模型定义了它，不能把它默认为 Stable。若以后某个更具体的 GenAI 文档或小节声明了不同状态，应以那个更窄的声明单独评估；在此之前，Development 是当前 GenAI 约定的控制信号。

在 GenAI span 中复用核心属性时，还要做两层检查。某个通用网络或错误属性可能已经在核心 registry 中稳定，但 GenAI span 的名称、requirement level、放置位置或解释仍可能处于 Development。单个属性稳定，不等于整个 signal 稳定。

### 2. 仓库状态不等于发布状态

核心 semantic-conventions 仓库在 v1.42.0 中把 GenAI 定义迁移到独立仓库，并废弃旧副本。这改变了所有权和事实来源，但没有把 GenAI 提升到 Stable。

独立仓库描述了未来的开发版发布通道：使用 `vX.Y.Z-dev` tag 和 `gen-ai-dev` schema URL。当前 releases 页面没有 release，changelog 没有已发布版本，README 也尚未公布 schema URL。因此：

- `main` 是持续变化的开发快照；
- open 或已经 merge 的 PR 都不等于已发布约定；
- 核心仓库旧 schema 的版本号不能证明迁移后的定义稳定；
- 自行编造看似官方的 GenAI schema URL 只会让迁移更困难。

等真实 tag 和 schema artifact 发布后再使用它们。在此之前，可以在部署元数据中记录精确源码 revision 或内部 profile 版本，但不要把它说成 OpenTelemetry 官方 schema release。

### 3. Instrumentation 稳定性和语义稳定性彼此独立

OTLP 可以携带任意属性。稳定的 SDK、exporter 或 instrumentation 包仍可能发出 Development 约定；两个包都可以输出合法 OTLP，却对 span 名称、字段要求或 value 语义持不同理解。

OpenTelemetry 的 telemetry stability 规则明确区分稳定和不稳定 producer。不稳定 producer 不保证其输出结构在版本间兼容。稳定 producer 必须明确标注这一承诺；在当前 schema transformation 暂停期内，也不能靠未来 schema 文件为破坏性输出变化兜底。

评估某个库时，要读它自己的 release notes 和稳定性声明。不能把 OpenTelemetry API、SDK 或 OTLP 协议的成熟度转移给它恰好发出的 GenAI 属性。

### 4. 提案成熟度不等于约定成熟度

Agent lifecycle events 是一个当前例子。公开提案定义了长时间运行 agent 的 pause、resume、checkpoint 和 pause resolution 事件，以及跨进程、跨 trace 的关联属性。设计已经很具体，也带有 reference scenario，但 PR 仍是 open 状态。这些名称仍是提案，不是已发布契约。

实验时可以在 feature flag 后发出它们，或在 collector 边界把内部事件映射到提议形状。对于必须跨升级存活的 dashboard、alert 或查询 API，应保留内部版本化 schema，等提案进入选定的正式 release 后再转换。

## 适合生产环境的采用方式

把每个 telemetry producer 当成实现了一个明确 profile。一个实用 profile 至少记录：

```text
instrumentation 包及版本
语义约定仓库 revision 或正式 schema URL
启用的 signal 与内容采集策略
collector normalization 版本
backend 查询和 dashboard 版本
```

然后建立一组小型 conformance fixtures：一次成功模型调用、一次流式调用、一次工具调用、一次 agent 或 workflow 操作，以及一次失败。每个 fixture 都断言 span 名称和 kind、operation name、模型/提供方标识、token 统计、错误表达，以及敏感内容默认是否缺失。保存期望的规范化输出，而不只是 dashboard 截图。

在 collector 边界：

1. 只接受已经测试的 producer profile；
2. 把它规范化到内部版本化 schema；
3. 对未知 profile 拒绝、隔离或单独标记，而不是静默混合；
4. 用低基数元数据保留来源 profile；
5. 导出前应用脱敏和基数控制。

升级时，让新旧版本重放同一组 fixtures，再 diff 规范化结果。若 dashboard 迁移需要重叠期，可以使用有界的 dual-read 或 shadow pipeline。不要对同一个 client library 同时运行两套 instrumentation；重复 span 和 token 计数会掩盖 schema 迁移是否正确。

包含内容的字段还要经过额外门禁。GenAI registry 明确警告 input/output message 属性可能含有敏感信息。即使字段将来 Stable，也不代表其内容安全、低基数或适合默认采集。稳定性、隐私和成本是三个独立的验收决定。

## 到什么程度才算“可以依赖”？

“依赖”可以有两种合理含义：

- **实验或边界明确的生产使用：**现在就可以，但必须固定 producer 版本和约定 revision，具备兼容性测试和内容采集控制，并让迁移层承担变化。
- **不希望迁移字段的长期公共契约：**应等相关 GenAI 文档或小节至少达到 Release Candidate，最好达到 Stable，同时采用正式仓库 release 和 schema artifact，而不是 `main`。

当前公开资料没有权威的逐属性毕业时间表。可靠信号是仓库状态、文档状态、正式发布 artifact，以及 instrumentation 自己的稳定性承诺。Issue、会议计划或被接受的设计能提示方向，但不能给出兼容日期。

## 参考资料

- [OpenTelemetry GenAI 语义约定：当前 Development 状态](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md)
- [OpenTelemetry GenAI 生成的属性 registry](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md)
- [OpenTelemetry 对 Development、Release Candidate、Stable 和 Deprecated 的定义](https://opentelemetry.io/docs/specs/otel/document-status/)
- [OpenTelemetry 对 telemetry producer 的稳定性要求](https://opentelemetry.io/docs/specs/otel/telemetry-stability/)
- [OpenTelemetry semantic-conventions v1.42.0：GenAI 定义迁移到独立仓库](https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.42.0)
- [GenAI 独立仓库：尚未发布 schema URL](https://github.com/open-telemetry/semantic-conventions-genai)
- [GenAI 独立仓库：没有正式 release](https://github.com/open-telemetry/semantic-conventions-genai/releases)
- [GenAI 独立仓库：当前 changelog 仍是 unreleased](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/CHANGELOG.md)
- [GenAI 独立仓库：计划中的开发版 release 和 schema 流程](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/RELEASING.md)
- [仍处于 open 状态的 agent lifecycle event 提案](https://github.com/open-telemetry/semantic-conventions-genai/pull/445)
- [Git submodule 初始化与递归更新](https://git-scm.com/docs/git-submodule)
- [Linux 中 `bpf_strncmp` helper 的实现](https://github.com/torvalds/linux/blob/master/kernel/bpf/helpers.c)
- [GNU C Library dynamic linker 文档](https://sourceware.org/glibc/manual/latest/html_node/Dynamic-Linker.html)
- [BPF 线程：在 trampoline image 释放前保留 program](https://lore.kernel.org/bpf/c145f1ec-a4fc-42e4-a267-0667775bf5f8@linux.dev/T/#t)
- [BPF 报告：释放 socket map 时任务挂起](https://lore.kernel.org/bpf/87ecfsvnb3.fsf@cloudflare.com/T/#t)
- [BPF 修复：单 CPU 的 per-CPU freelist 无限循环](https://lore.kernel.org/bpf/178725180892.429815.3721455388690610183.git-patchwork-notify@kernel.org/T/#t)
- [BPF 修复：在有符号迭代溢出前拒绝过大的 array map](https://lore.kernel.org/bpf/20260820084643.35489-1-meishaoming@xiaomi.com/T/#t)
- [bpftool 修复：排序后的 C dump 不再漏掉 type](https://lore.kernel.org/bpf/478fbefa108d7da8eb28897998857ab72467b276.camel@gmail.com/T/#u)
- [cgroup 提案：通过 BPF kfunc 暴露 CPU 统计](https://lore.kernel.org/bpf/20260818002450.3071325-1-ziyang.meme@gmail.com/T/#t)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自过去 24 小时，因此没有使用七天 fallback。姓名、账号、雇主、workspace 和频道身份、封闭聊天链接、精确时间、私有拓扑、原始日志及可搜索回原讨论的措辞均已删除，也没有保留原始 transcript。

### GenAI 使用者需要兼容性信号，而不只是一张属性清单

最主要的可观测性讨论追问：在哪里能看到 GenAI 属性的当前状态和预期时间表。另一个相关讨论提出了 agent pause、checkpoint 和 resume 的 lifecycle events。共同的运营问题是：dashboard、指标聚合和持久查询需要知道字段是否是长期契约，而 instrumentation 作者又需要空间纠正尚未完成的模型。

当前做法应是把 GenAI 整体视作 Development，把 open proposal 视作实验能力。[官方状态定义](https://opentelemetry.io/docs/specs/otel/document-status/)和[当前 GenAI 总览](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md)是权威成熟度信号；[仍未合并的 lifecycle 提案](https://github.com/open-telemetry/semantic-conventions-genai/pull/445)只说明演进方向。实用诊断路径是盘点 producer 包和实际输出 profile，再与固定 revision 的 registry 对照。尚未解决的是毕业时间：当前公开资料没有给出逐属性日期。

### 一些“eBPF 构建失败”实际发生在 eBPF 开始运行之前

几条项目支持信号看似是 eBPF loader 错误，实际属于更早的层。克隆后缺少 libbpf 或 bpftool header，应先检查仓库 submodule 是否初始化；[Git submodule 契约](https://git-scm.com/docs/git-submodule)说明 `update --init --recursive` 会填充 superproject 记录的 commit。Verifier 报告涉及不可用 helper 时，应对照正在运行内核的 helper 集，而不仅是编译时源码树；当前[内核 helper 实现](https://github.com/torvalds/linux/blob/master/kernel/bpf/helpers.c)也说明 `bpf_strncmp` 是内核提供的 helper，不是 userspace fallback。

另一个 warning 来自系统级 preload 配置触发的 ELF dynamic loader，而 BPF 程序本身仍能运行。[GNU C Library 文档](https://sourceware.org/glibc/manual/latest/html_node/Dynamic-Linker.html)明确了所有权边界：dynamic linker 在应用启动前加载依赖。因此合理诊断顺序应是仓库依赖、可执行文件 loader、内核/helper 可用性、libbpf open/load，最后才是 verifier 或 attach。仍不确定的是版本相关 packaging：source build 可以改变 launcher 路径，但不会修复 host-wide preload 配置。

### 内核讨论集中在生命周期、销毁路径和有界迭代

公开内核工作反复追问：当另一个内核对象仍可能引用某资源时，谁必须继续存活。一项修复让 BPF program 保持存活，直到调用它的 trampoline image 释放；另一份报告发现 socket-map teardown 在 socket lock 内等待导致任务挂起。这些都是失败路径问题：成功 attach 或插入不能证明 detach、replace 和 destroy 使用了相同所有权顺序。公开证据是[trampoline 生命周期修复](https://lore.kernel.org/bpf/c145f1ec-a4fc-42e4-a267-0667775bf5f8@linux.dev/T/#t)和[socket-map teardown 报告](https://lore.kernel.org/bpf/87ecfsvnb3.fsf@cloudflare.com/T/#t)。

另外两项较小修复从数值边界暴露了同一教训：只可能有一个 CPU 时，per-CPU freelist 可能无限循环；array map 接受过大尺寸后，后续有符号 iterator 会溢出。回归矩阵必须覆盖最小拓扑、最大允许尺寸、部分初始化、rollback 和并发销毁。[单 CPU freelist 修复](https://lore.kernel.org/bpf/178725180892.429815.3721455388690610183.git-patchwork-notify@kernel.org/T/#t)与[array 大小保护](https://lore.kernel.org/bpf/20260820084643.35489-1-meishaoming@xiaomi.com/T/#t)仍属于活跃内核改动，运营者应核对实际交付内核，而不能把 upstream patch 当成已经部署。

### 可观测接口在扩展，但输出本身仍要做正确性测试

工具链工作修复了一个排序后的 bpftool C dump 静默漏掉 type 的问题。这比语法错误更危险，因为生成结果可能看起来合理却不完整。测试应比较排序前后完整的 type identity 集合，而不仅是编译一份生成 header。[bpftool patch](https://lore.kernel.org/bpf/478fbefa108d7da8eb28897998857ab72467b276.camel@gmail.com/T/#u)为这一边界补充了覆盖。

另一项提案通过 kfunc 向 BPF 暴露 CPU cgroup 统计，让内核内策略不必从无关事件重建 accounting 数据。但读取接口本身并不定义调度或 enforcement policy，调用方仍需处理 hierarchy、counter 语义和内核版本可用性。[公开 cgroup 系列](https://lore.kernel.org/bpf/20260818002450.3071325-1-ziyang.meme@gmail.com/T/#t)仍在 review 中，不是可移植的已发布接口。

其余项目、调度、网络和公开论坛目标在当日窗口内没有新的实质技术条目，或只有更早讨论和例行通知。它们全部可访问，并被记录为安静，而不是跳过。
