# 被 OBI 插桩的 Node.js 服务，为什么不能因为不用手动 span 和日志富化就直接关掉每次回调的 `async_hooks` trace 上下文哨兵？

**简短回答：** 因为这个哨兵不是 OBI 自己的功能，而是一个共享的、被 pin（按名字固定）的内核 map 的写端，它的消费者集合超出了 OBI 自身的配置范围。注入的 Node 智能体的 `async_hooks` `before` 钩子（在每个 JS 回调之前触发 `fs.existsSync("/dev/null/obi-ctx/<fd>")`）让 `traces_ctx_v1` —— 一个按名字固定的 LRU hash，键是 pid/tgid，值是活动的请求的 `{trace_id, span_id}` —— 与该线程上正在执行的请求保持一致。OBI 自己的消费者（手动 span 父化、日志 trace 标注）只是消费者集合的一部分：这个 map 的头部声明它的规格属于一个 OTEP，并警告修改它"可能破坏依赖它其他组件"；一个读这个被 pin 的 map、把它的 profile 关联到 OBI 的 trace 的进程外 eBPF profiler，就是 OBI 的配置选择器看不到的那个消费者。所以一个按"手动 span 关 + 日志标注关"算出来的 gate 看起来是完整的——却悄悄丢掉了外部 trace/profile 关联这条通道。

## 哨兵实际维护的是什么

成本故事（每次回调的哨兵就是注入智能体开销所在，而不是 eBPF 探针）已经是 p99 问题的答案；本文是另一个边界——*关闭它的安全性*。

智能体侧（`pkg/internal/nodejs/fdextractor.js`），trace 上下文传播机制被刻意做成昂贵的那部分："net 原型包装 + 每次回调都触发的 async_hooks before 钩子"，而且智能体用 `TRACES_ENABLED` 模板化，让纯 metrics 注入"完全跳过"它。`before` 钩子在每个 JS 回调之前触发，并通过固定路径上的 `fs.existsSync` 向内核发信号——在 `async_hooks` 内是安全的，因为同步 fs 操作不会产生 `AsyncWrap`，也就不会重新触发这个钩子。有两种哨兵形式：

- `/dev/null/obi-ctx/<fd>` —— 刷新：当前 async 上下文的 4 位 incoming fd，让内核 map 反映活动请求；
- `/dev/null/obi-noreqctx` —— 清除：在请求到非请求的转换时发出（后台定时器、请求结束后才跑的回调），避免稍后的 span 被挂到前一个请求的 trace 下。

内核侧（`bpf/generictracer/nodejs.c`），这些路径由同步 fs 调用上的 uprobe 处理器解码：`handle_async_switch` 刷新 map，`handle_ctx_clear` 删除条目，`handle_node_span` 读它。map 本身（`bpf/shared/obi_ctx.h`）是：

```c
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __type(key, u64);                 // pid/tgid
    __type(value, obi_ctx_info_t);    // { trace_id, span_id }
    __uint(max_entries, 1 << 14);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} traces_ctx_v1 SEC(".maps");
```

它上面有那句决定性的注释："NOTE: this map spec is part of an OTEP (open-telemetry/opentelemetry-specification#4855). Changing its spec may break other components relying on it."（注意：这个 map 规格属于一个 OTEP。修改它可能破坏依赖它的其他组件。）

## 到底谁在读 `traces_ctx_v1`

三类消费者都重要：

1. **手动 span。** `spanbridge.js` 通过 `/dev/null/obi-span/<json>` 发出手动 span 的结束；内核的 `handle_node_span` 用 `obi_ctx__get(pid_tgid)` 给事件盖上父 id，让手动 span "可以挂到 OBI 的自动 server span 下"。没有每次回调的刷新，结束在 async 回调里的手动 span 就会带上过时的或缺失的父节点。
2. **日志 trace 标注。** OBI 的配置 schema 暴露了一个日志 trace 标注块，携带 `trace_id`/`span_id` 字段名（`internal/config/schema/correlation.go`）。对一个被插桩的 Node.js 进程，一条日志被归到该线程的 trace 上下文，正是哨兵在 map 里维护的按线程上下文——所以只有在哨兵让 map 与活动请求保持一致时，标注才保持正确。
3. **外部 trace/profile 关联。** 把 OBI trace 关联到 profile 的 OTEP（open-telemetry/opentelemetry-specification#4855；PoC 在 OBI PR #1184 与 Coralogix eBPF profiler）正是 map 头部声明规格所属的那个 OTEP。一个读被 pin 的 map、用它自己的 profile 样本去 join OBI trace id 的进程外 eBPF profiler 就是这第三个消费者。已合入的进程上下文 OTEP（`oteps/profiles/4719-process-ctx.md`）是更宽的 spec 级通道，供进程外 eBPF profiler 读每进程上下文。

## 为什么 OBI 的 config gate 看不到第三个

人们会抓的那个 gate 完全用 OBI 的功能词汇来表达："手动 span 关、日志标注关，所以 trace 上下文机制用不到——跳过它。"这正是 `TRACES_ENABLED` 开关，也是合理的*成本* gate。但它按构造欠完整，因为 map 的消费者集合不是由 OBI 的功能开关定义的。pin（`LIBBPF_PIN_BY_NAME`）就是存在进程外读者的信号：一个不参与 OBI 配置的组件通过 pin 读 map，而没有任何 OBI 配置选择器说"但要把上下文为外部关联集成留着"。当你只按 OBI 自己的消费者来算 gate，结果*看起来*完整——OBI 的手动 span 和日志标注都关了，OBI 内部不会有任何抱怨——而外部 trace/profile 关联通道无声地熄灭，OBI 里任何地方都没有错误。

## 如何安全地 gate

1. **枚举 map 的消费者，而不是只枚举 OBI 的功能。** 被 pin 的 `traces_ctx_v1` 有手动 span、日志标注、外部关联三类消费者。任何关掉哨兵的 gate 都必须考虑你部署里读这个 map 的每个消费者。
2. **只要有进程外 profiler 关联到 OBI 的 trace，就保留哨兵。** 如果你跑一个把 profile 样本 join 到 OBI trace id 的 eBPF profiler（trace/profile 关联 OTEP 的集成），那 profiler 读到的上下文就是哨兵维护的——无论 OBI 功能开关如何，都要让这个 gate 保留。
3. **只有在没人读 map 时才用纯 metrics 路径。** `TRACES_ENABLED=false` 的纯 metrics 注入完全跳过 trace 上下文传播——当没有消费者（手动 span、日志标注、外部关联）需要这个 map 时是对的，只要有其中一个需要就是错的。
4. **验证通道本身，而不是没有错误。** 改完 gate 之后，确认手动 span 仍然挂在自动 server span 下，外部工具的 profile-to-trace join 仍然有效。关联消费者失败在 OBI 里不产生错误；症状是外部工具关联里的缺口。

## 决定它的边界

边界是*一个被 pin、被规格指定的 map 的消费者集合由谁拥有*。因为 map 是按名字 pin 的、其规格被声明属于一个 OTEP——一个跨组件契约——它的消费者超出 OBI 的配置面。只按 OBI 功能开关算的 gate 能表达"没有 OBI 消费者"，却不能表达"没有任何消费者"，而第三个消费者恰恰活在 OBI 的配置之外。实用规则：按消费者集合来 gate，而不是按功能开关；把 map 的 pin 当作存在外部读者的信号，只要有外部读者就保留哨兵。

## 参考资料

- [OBI — `bpf/shared/obi_ctx.h`（被 pin 的 `traces_ctx_v1` map：LRU hash，键为 pid/tgid，值为 `{trace_id, span_id}`，`LIBBPF_PIN_BY_NAME`；"this map spec is part of an OTEP … Changing its spec may break other components relying on it"）](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/bpf/shared/obi_ctx.h)
- [OBI — `bpf/generictracer/nodejs.c`（哨兵解码器：`handle_async_switch` 刷新 map，`handle_ctx_clear` 删过时条目，`handle_node_span` 经 `obi_ctx__get` 给手动 span 挂父）](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/bpf/generictracer/nodejs.c)
- [OBI — `pkg/internal/nodejs/fdextractor.js`（智能体侧的 `async_hooks` `before` 钩子、`obi-ctx`/`obi-noreqctx` 哨兵、以及"skipped entirely for metrics-only injections"的 `TRACES_ENABLED` 成本 gate）](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/pkg/internal/nodejs/fdextractor.js)
- [OBI — `internal/config/schema/correlation.go`（携带 `trace_id`/`span_id` 字段名的日志 trace 标注配置块）](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/internal/config/schema/correlation.go)
- [OTEP: correlating OBI traces to profiles（spec PR #4855，已关闭未合入；PoC 在 OBI PR #1184 与 Coralogix eBPF profiler）](https://github.com/open-telemetry/opentelemetry-specification/pull/4855)
- [OTEP — Process Context: Sharing Resource Attributes with External Readers（已合入的 spec 级进程外 eBPF profiler 通道；注明 OBI 适用性）](https://raw.githubusercontent.com/open-telemetry/opentelemetry-specification/main/oteps/profiles/4719-process-ctx.md)
- [Node.js — `async_hooks` API（`before` 钩子在每次 async 操作对应回调之前触发；同步 fs 操作不产生 `AsyncWrap`）](https://nodejs.org/api/async_hooks.html)

## 当日社区讨论

本监控窗口是跨两个已 opt-in 只读归档的 Slack 归档（均为 OpenTelemetry 插桩频道）的滚动一周；两个可见浏览器聊天工作区与公共邮件列表/子版表面本次未审阅，该缺口已如实记录而不视为安静。四条主题自前几日的窗口延续而来，且均已发布：OBI 配置 v1 到 v2 迁移契约、GenAI 智能体 span 形状、托管智能体 harness 可观测性边界、Node.js 哨兵成本与智能体生命周期。本轮切片里唯一实质新颖、有源码依据的边界，就是上文的 Node.js 哨兵 gate 问题。

**Node.js 哨兵 gate 与隐藏消费者（即上文问题）。** 反复出现的 Node.js 成本线——eBPF 探针便宜，注入智能体每次回调的哨兵才是开销——引出了本文回答的后续问题：当某部署不用手动 span 或日志富化时，哨兵能否被 gate 掉。工作答案是消费者集合边界：哨兵是一个被 pin、被 OTEP 指定的 map 的写端，其消费者集合超出了 OBI 自己的功能开关。维护者确认的读法（已匿名化）是：客户端 span 父化来自 fd 对 map，每次回调的哨兵让 trace 上下文 map 与活动请求保持一致，而第三个消费者（外部 trace/profile 关联）也读这个被 pin 的 map——所以只按手动 span 与日志富化来 gate 哨兵，可能悄悄破坏那个集成。该读法的公开源码依据是 map 头部的 OTEP 契约注释、`handle_node_span` 的父查找、以及 trace-to-profile OTEP 与其 profiler PoC；安全规则是按 map 的消费者集合来 gate，把 pin 当作存在外部读者的信号。
