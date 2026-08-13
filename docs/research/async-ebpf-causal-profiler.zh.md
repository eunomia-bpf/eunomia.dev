---
date: 2026-08-13
title: "eBPF 异步性能分析为什么会丢失因果关系？"
description: "eBPF 栈采样能看到 CPU 时间花在哪里，但异步任务跨线程、io_uring 和工作队列后会丢失请求因果链。本文比较现有机制，并提出可验证的因果性能分析设计。"
tags:
  - Daily Report
  - eBPF
  - 性能分析
  - io_uring
  - 异步运行时
research_question: "eBPF profiler 还需要哪些运行时证据，才能在 scheduler wakeup、workqueue、io_uring、用户态异步任务和应用自定义资源之间重建因果关系，同时又不把持续性能分析变成全量 tracing？"
source_cutoff: 2026-08-13
status: daily-report
---

# eBPF 异步性能分析为什么会丢失因果关系？

CPU profiler 很擅长回答一个问题：采样发生的这一刻，机器正在什么调用栈里花时间？但一旦一个请求不再由最初接收它的线程继续执行，这个问题就不够了。

想象一个请求先在线程 A 解析输入，然后提交一个 `io_uring` 操作，唤醒某个 runtime task，把一部分工作放进内核 workqueue，之后在线程 B 恢复，最后又等待应用自己维护的 buffer pool。普通栈采样可以分别看到这些执行片段，却不会自动知道它们属于同一个请求，也不知道哪一次等待导致了后面的哪一段执行。如果真正昂贵的代码跑在通用 worker 上，最明显的火焰图甚至只会告诉你 worker 在忙，而不是谁把这份工作交给了它。

<!-- more -->

这并不说明 stack sampling 已经过时。[OpenTelemetry eBPF Profiler](https://github.com/open-telemetry/opentelemetry-ebpf-profiler) 已经证明，eBPF 可以在不注入目标进程的情况下做跨语言、全系统、持续的栈采样。OpenTelemetry 的 [Profiles specification](https://opentelemetry.io/docs/specs/otel/profiles/) 也给出了清晰的数据模型，而且在已有 trace context 时，profile sample 可以直接带 trace/span link。

真正缺的是另一层信息：**异步 profiler 除了 sampled stack，还需要一张有边界的 causal handoff graph**。它只记录那些会把责任从一个执行上下文交给另一个上下文的事件，例如 enqueue、submit、wake、start、complete、resume。之后，普通栈采样再挂到最近的执行身份上，并沿这些 handoff edge 做归因。如果内核根本看不到语义身份，就需要一个很小的 runtime join contract，而不是假装 OS profiler 能从 syscall 里猜出所有 Future、goroutine、请求和应用资源。

这里的目标刻意小于 full tracing。应用正确接入 tracing 后，本来就可以保存请求上下文。值得研究的问题是：**能不能用接近持续 profiling 的开销和部署方式，恢复足够多的因果结构来做性能诊断？**

本文延续前面的 [用户态 eBPF runtime contract](https://eunomia.dev/zh/research/userspace-ebpf-runtime-contract/)、[hook composition contract](https://eunomia.dev/zh/research/ebpf-hook-composition-contract/) 和 [有状态升级报告](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)。前几篇讨论程序怎样 attach、怎样组合、怎样升级。这一篇把同样的 runtime 视角放到 profiler 上：如果 observation 没有稳定身份和生命周期规则，就很难把不同位置采到的证据组合成一个因果解释。

## 栈采样能定位执行位置，却不天然保存 handoff 身份

sampling profiler 的优势来自稀疏性。它不记录每一次调用，而是周期性观察正在运行的上下文，用样本估算执行时间分布。OpenTelemetry 把 profile 定义成 stack trace 和对应的 value；它的 eBPF profiler 则把重点放在无需目标进程 instrumentation 的 whole-system stack collection。

问题也来自同一个地方。一个 sample 只描述一个观察点，并不保存“这份工作为什么会变成 runnable”的历史。

async runtime 会把这个边界放大。Tokio 的 [`spawn`](https://docs.rs/tokio/latest/tokio/task/fn.spawn.html) 文档明确说明，spawn 出来的 task 可能在当前线程运行，也可能被送到另一个线程。Rust `tracing` 因而提供 [`Instrument`](https://docs.rs/tracing/latest/tracing/trait.Instrument.html)，把 span 绑定到 Future，并在 Future 每次被 poll 时进入该 span。这个显式传播机制本身就说明：当任务可以迁移时，thread-local context 不是稳定的 task identity。

Go 用另一种方式暴露了相同问题。[`runtime/trace`](https://pkg.go.dev/runtime/trace) 把 task 定义成一个可能由多个 goroutine 共同完成的逻辑操作，并通过 `context.Context` 传递 task identity。Go runtime 能做这件事，是因为它知道 goroutine 和用户 annotation。一个通用的内核 sampler 并不知道。

所以 async profiler 至少要把三个概念拆开：

1. **Execution location**：stack、process、thread、CPU 和 timestamp。
2. **Causal handoff**：把责任从一个执行上下文转移到另一个上下文的事件。
3. **Semantic ownership**：request、task、Future、resource 或 operation，这些身份可能根本不在 kernel object 里。

把三者都塞进 stack 会丢信息。为了补信息而把所有事件都 trace 下来，又失去了 profiling 的意义。

## Linux 已经暴露了一些很有价值的因果边

好消息是，几个重要的 async boundary 已经有可以关联的 kernel field。profiler 并不需要靠“两个事件时间很接近”去猜所有关系。

### Workqueue 在入队和执行两端都暴露 work object

Linux 的 [`workqueue` tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/workqueue.h) 在 `workqueue_queue_work` 记录 work 入队，又在 `workqueue_execute_start` / `workqueue_execute_end` 记录 worker 真正执行。两端都带 `work_struct` pointer 和 function pointer。

于是可以形成一条直接的关联：

```text
producer thread
    |
    | queue(work_struct = W)
    v
work item W
    |
    | execute_start(W)
    v
worker thread + sampled stack
```

stack-only profiler 往往会把后面的 CPU 时间算给通用 worker。edge-aware profiler 则可以把 worker execution 重新连回最初 enqueue `W` 的 producer。

当然，pointer 不是永久全局唯一 ID。kernel object 会复用。因此真正的 collector 不能永远保存裸 pointer，需要 sequence、owner scope、时间窗口或 generation guard 来约束生命周期。

### io_uring 已经带着 request、ring 和用户 correlation token 走完整个异步边界

Linux 的 [`io_uring` tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/io_uring.h) 更直接。`io_uring_submit_req`、`io_uring_queue_async_work` 和 `io_uring_complete` 会暴露 ring context、request pointer、opcode、`user_data` 等字段。UAPI 和 [liburing header](https://github.com/axboe/liburing/blob/master/src/include/liburing/io_uring.h) 规定 SQE 的 `user_data` 会在 completion 时返回，liburing 还提供 helper，把一个 pointer 或 64-bit value 放进 SQE，再从 CQE 取出来。

这几乎已经是一条现成的 handoff key。profiler 可以在 submit 时抓住仍然有 submitter stack/context 的那一刻，保存一个有生命周期限制的 `(ring, user_data, generation)` 关系，然后在 completion 或 async worker 上重新关联。

但 `user_data` 属于应用。它可能是 pointer、counter、packed ID，也可能重复使用。profiler 不能直接把它当 trace ID。multishot operation 更说明生命周期不能靠猜。正确的理解是：**kernel 帮应用把一个自选 correlation token 穿过了 async I/O boundary，但 token 的语义和唯一性仍需要 owner/lifetime contract。**

### Scheduler wakeup 告诉我们“谁唤醒了谁”，却不一定告诉我们“为什么”

Linux [`sched` tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/sched.h) 提供另一类部分因果边。`sched_waking` 文档说明它从 waking context 执行，同时给出被唤醒 task；`sched_switch` 则告诉我们哪个 task 离开或进入 CPU。

因此 profiler 可以建立“当前 execution 唤醒 target task”的 edge，并在之后计算 run-queue delay。但同一个 shared event-loop 可能服务很多请求，所以 scheduler causality 仍然没有 request semantics。它是重要证据，不是完整答案。

## Profile format 能保存已有的 trace context，却不能凭空生成它

OpenTelemetry Profiles 的一个很合理的设计是：sample 可以带一个包含 trace ID / span ID 的 `Link`。因此新的 eBPF causal profiler 没必要发明另一个应用 tracing 格式。如果采样时已经知道合法 span context，直接 link 就好。

困难的恰恰是 context 不存在或者在某个 boundary 断掉的场景。closed-source service 可能根本没接 tracing SDK；kernel workqueue callback 不会自动继承 userspace span；`io_uring` completion 可能在 submitter stack 已经消失后由 shared thread 处理。profile schema 可以存 link，但 schema 本身不会替你创造 link 的含义。

因此一个更可互操作的规则是：**有 trace/span ID 时就把它当一等 semantic identity；没有时，用 eBPF handoff edge 延长 attribution，但不要伪造 span。** 这样 profiler 是 OpenTelemetry 的补充，而不是另一套平行 observability universe。

## Application-defined resource 是另一道语义边界

OSDI 2026 的 [gigiprofiler](https://www.usenix.org/conference/osdi26/presentation/hu-yigong) 是一个很有用的反例。它处理 buffer pool、query cache、temporary structure 这类 system metrics 不理解的应用自定义资源。它先用语义推断和 static analysis 找候选资源与 usage event，再在运行时追踪 request 如何使用这些资源。论文在五个真实应用的 15 个性能问题上都完成了检测和诊断，并另外发现两个 MariaDB 问题，之后被开发者确认。

这里对 eBPF profiler 的启发不是“也加一个 LLM”。真正的结论是：**有些 performance state 根本没有 kernel-native 名字。** 如果问题是“哪个 request 占着这个 buffer pool entry”，只有 syscall trace 并不够，除非 profiler 能把 application resource identity join 到 request 和 execution context。

所以通用 async profiler 更适合采用分层策略：

- kernel 已经暴露 correlation object 时，直接用 tracepoint；
- runtime 有可观测 task identity 时，用 uprobe 或稳定 runtime hook；
- application-defined resource 没有外部稳定观察点时，接受一个很小的 annotation 或 USDT join point；
- 实在拿不到 edge 时，就保留 stack-only attribution，并明确标成 unknown parent，不要靠时间邻近猜关系。

“unknown parent” 比一个自信但错误的因果边更有价值。

## 自适应 capture 会让 sampling bias 变成系统设计问题

continuous profiling 必须有 overhead budget。如果把每个 `sched_waking`、每个 workqueue event、每个 `io_uring` request、每次 runtime poll 和高频 stack sample 都保存下来，数据量会越来越接近 tracing。真正可部署的系统很可能要 selective capture 或 adaptive capture。

但 selection 会改变 estimator。

OSDI 2026 的 [Blink](https://www.usenix.org/conference/osdi26/presentation/devsot) 提醒我们，sampling error 不一定只是方差变大。对于大量短函数、profile 很平的 mobile workload，论文报告 `perf` 一类 sampling profiler 会受到 skid、shadow effect 和 function coverage 不完整影响，从而产生系统性错误。Blink 改用轻量 instrumentation，在其评测 workload 中报告 99.999% accuracy 和 1% overhead。

async causal profiling 还多了一种 selection bias。假设 profiler 看到一次很长的 `io_uring` wait 后，临时把那一类 request 的采样率提高十倍。如果之后直接按 raw sample count 做聚合，slow request 就天然比 normal request 更容易被看到。系统可能真的找到了异常样本，却把“这类路径占全部 CPU 的百分比”算错了。

因此设计上应该加一条明确规则：**只要 capture probability 会根据已观察行为变化，就把有效 inclusion probability 和 sample/edge 一起保存。** 之后才有可能用 weighted estimator 恢复总体量。如果 selection policy 根本说不清概率，就只能把结果叫 targeted diagnostic evidence，不能把它包装成 unbiased profile。

## Bounded handoff graph 比 universal trace 更适合作为中间层

一个实际的数据面不需要保存完整 request history，只要存足以跨越 execution identity 的边。

例如可以定义：

```text
HandoffEdge {
    source_execution
    target_kind
    target_id
    target_generation
    edge_kind          // wake, queue, submit, complete, spawn, poll ...
    timestamp
    expires_at
    sample_probability
}
```

`source_execution` 可以是 process/thread 加当前已知 semantic context；`target_id` 可以是 work object、io_uring token、runtime task 或 application resource；`target_generation` 用来避免 pointer reuse 把两个生命周期错误合并。等 worker 开始执行或 runtime task resume 时，target 又变成新的 source。

这张图必须在四个方向上有界：

- **time**：operation 生命周期结束后过期；
- **scope**：只对目标 process/cgroup/workload 开启；
- **edge type**：只打开当前诊断需要的 handoff family；
- **rate**：高频 edge 要按已知概率采样，或者只在一个有限诊断窗口里 exact capture。

它和 distributed tracing 的目标不同。trace 通常希望保留一个完整的 end-to-end request history；profiler 只需要临时 lineage，把 sparse sample 和 wait 归因。聚合完成后，大部分 raw edge 都可以丢掉。

## 现有研究还缺什么

### Zero-instrumentation profiler 仍然会停在 runtime semantic boundary

whole-system eBPF profiler 可以在不改目标应用的情况下 unwind 很复杂的 mixed stack，但它不会自动知道不同 Tokio worker 上的两次 poll 属于同一个 Future，也不会知道三个 goroutine 是在协作完成同一个逻辑 task，除非 runtime identity 有稳定外部表示。

缺的是一个 portable、足够小的 runtime task join contract。今天的实际选择往往是：要么接受 low-overhead stack profiling 但丢 semantic lineage，要么接更丰富的 tracing，并让 runtime/application 参与。

真正有区分度的实验是：一个小 adapter 能否恢复大多数 request attribution，同时明显比 full tracing 更低成本、更容易部署？如果不能，它只是换了名字的 tracing SDK。

### Kernel correlation ID 的 lifetime semantics 并不统一

`work_struct *`、io_uring request pointer、`user_data`、PID、runtime task ID 都有不同复用规则。把任何一种 ID 永远当 causal identity，都可能在对象复用后产生 false join。

缺的是显式 identity model：object kind、owner scope、generation、retirement condition 应该一起决定 identity。否则 profiler 表面上看起来“链路完整”，实际上把两个无关 operation 接起来，反而更危险。

应该专门做 object reuse stress test。如果一个简单 TTL 加 owner tuple 就能在真实 workload 中消灭 false join，就没必要发明更重的 identity protocol。

### Profiles 可以 link traces，但缺少通用 low-level handoff vocabulary

OpenTelemetry Profiles 能把 sample link 到 span；Linux tracepoints 能给 queue/complete edge。两者之间还缺一个小而通用的 vocabulary，描述 execution A 提交 resource R、R 让 execution B resume、B 正在等待 application resource X。

如果没有这个层，各个 profiler backend 会继续自己发明 join rule，也分不清“这里确实没有 causal edge”和“collector 不支持这类 edge”。

最关键的评测是 portability。同一套 handoff schema 应该能表示 workqueue、io_uring、Tokio、Go 和一个 application-defined resource，同时又不抹掉它们不同的生命周期规则。

### Adaptive causal capture 可能更会找问题，却让 aggregate 失真

fixed-rate stack sampler 的统计含义相对熟悉。但如果 profiler 一看到可疑 wait 就提高那一小部分 execution 的 edge/sample rate，进入数据集的概率已经改变。

缺的是 probability-aware accounting，以及 diagnostic evidence 和 population estimate 的明确区分。否则系统可能真的找到一个 slow request，却把“它占全部成本多少”说错。

最有价值的 evaluation 不是再画一张 overhead 曲线，而是做 coverage calibration：profiler 说某条 causal path 占 20% CPU 或 blocked time 时，高保真 ground truth 是否真的落在报告的 confidence interval 里？

## 同时具有学术价值和生产价值的方向

### 1. 为 kernel-visible async boundary 做 causal handoff ledger

**Gap。** work 跨 thread、worker pool、async I/O 后，stack sample 丢失 lineage，但 Linux 在若干 boundary 已经暴露 correlation object。

**Mechanism。** 在选定的 workqueue、io_uring、scheduler 和 syscall tracepoint 上挂 eBPF，把 queue/submit/wake 与 start/complete 统一成 bounded handoff graph。join key 使用 `(scope, object, generation)`，而不是裸 pointer。之后的 stack sample 和 off-CPU interval 沿 graph 归因，聚合结束就回收 edge。

**Delta。** 现有 eBPF continuous profiler 主要解决 stack collection 和 unwinding；full tracer 保存更丰富的完整历史。这里仅记录跨 execution context 所必需的边。

**Artifact。** 一个 libbpf collector、一套 compact handoff schema、online join engine，以及 pprof/OpenTelemetry Profiles exporter。已有 trace context 时保留 trace link，没有时输出明确的 causal label 或 unknown edge。

**Evaluation。** 构造 synchronous、workqueue、`io_uring`、scheduler wakeup 和混合 handoff workload，并用受控 instrumentation 生成 ground truth。测 causal-edge precision/recall、sample attribution accuracy、ID reuse 下的 false join、state size、event rate、CPU overhead。基线包括 stack-only profiler 和 full tracing。

**Academic value。** 可以验证 causal reconstruction 是否真的存在于 profiling 和 tracing 之间，成为一个独立的 sparse graph 问题。

**Production value。** operator 可以回答“这段 worker CPU 到底是哪个 submitter/request 造成的”，而不需要所有服务永久打开 full tracing。

**Failure condition。** 如果 exact edge capture 很快逼近 trace-level overhead，或者 object reuse 让外部 join 无法可靠完成，这个中间层就没有部署优势。

### 2. 把 adaptive profiling 变成 probability-aware measurement

**Gap。** profiler 必须守住 overhead budget，但围绕异常动态提高 sample/edge rate 会改变样本进入数据集的概率，导致 aggregate bias。

**Mechanism。** 使用 randomized base sampling，再让 controller 按 scope 或 edge family 改变 capture probability。每条 sampled observation 保存有效 inclusion probability。CPU、wait 和 handoff contribution 用 inverse-probability weighting 估计，并给 uncertainty。无法定义 sampling probability 的 targeted capture 单独标记，不进入无偏总体估计。

**Delta。** adaptive profiler 往往只优化“下一个 sample 花在哪里”。这个方向把 selection policy 也纳入 measurement contract，使调整 rate 不会悄悄改变百分比的含义。

**Artifact。** sampling controller、handoff schema 中的 probability metadata、weighted aggregator，以及包含 synthetic phase change 和真实 async service 的 calibration harness。

**Evaluation。** 比较 fixed-frequency sampling、randomized sampling、naive adaptive、probability-corrected adaptive，以及高保真 tracing/instrumentation。测 error、confidence-interval coverage、detection latency、overhead、variance。加入类似 Blink 的大量短函数 workload，要求系统能识别“这里 sampling 本身就不适合作为真值来源”。

**Academic value。** 把 profiler control policy 和 statistical estimability 联系起来，而不是只用 hotspot detection rate 评价 adaptivity。

**Production value。** 系统可以把更多 budget 花在刚出现的异常上，同时不让 targeted capture 伪装成 fleet-wide unbiased profile。

**Failure condition。** 如果 inverse-probability weighting 在真正有用的 adaptive policy 下方差太大，就应该放弃 population estimate，只把 adaptivity 用于 case finding。

### 3. 给 opaque runtime task 和 application resource 定义最小 join contract

**Gap。** kernel 不知道所有 Future、goroutine task、request-local cache entry 和 application-defined resource。full tracing 能带这些语义，但要求所有应用接 tracing SDK，会直接失去 zero-instrumentation 的优势。

**Mechanism。** 定义可选 join ABI，只保留几个操作：创建 semantic ID，把 semantic ID handoff 到 runtime/resource ID，activate/deactivate，retire。实现可以用稳定 runtime hook、uprobe、USDT 或很小的 library call。已有 OpenTelemetry trace/span ID 时直接导入，不采集任意 application payload。

**Delta。** Go `runtime/trace` task 和 Rust `tracing::Instrument` 说明 async semantic context 可以传播；gigiprofiler 则说明 request-to-resource attribution 有实际诊断价值。这里的 contract 刻意比完整 language trace 或 resource-specific profiler 更小。

**Artifact。** versioned C ABI/schema，Tokio 和 Go reference adapter，一个 application-defined resource demo，以及能把 adapter identity 和 kernel handoff edge 合并的 eBPF collector。

**Evaluation。** 同一个服务跑四种模式：stack-only eBPF、kernel handoff graph、handoff graph + join adapter、full application tracing。比较 semantic attribution coverage、正确率、setup effort、event volume、overhead、runtime version robustness。

**Academic value。** 能找出 causal profiling 至少需要多少 semantic information 穿过 userspace/kernel boundary，并明确 zero-instrumentation 的真正终点。

**Production value。** 默认仍然做 whole-system profiling，只对那些确实卡住诊断的 runtime 或 resource 增加小 adapter，而不是把所有应用一次性改造成 tracing application。

**Failure condition。** 如果 adapter 依赖不稳定 internal symbol、每次 runtime 升级都要维护，或者改动和 overhead 已经接近标准 tracing，那么直接用现有 tracing 才是更好的 interface。

## 真正的评测应该测 attribution，而不只是测 overhead

一个新 profiler 很容易在“CPU overhead”和“data volume”两张图上看起来漂亮。但这里的核心 claim 是 causal attribution，所以必须有 ground truth。

可以用下面的 workload matrix：

| Workload | Handoff | Ground truth | 最主要要发现的错误 |
| --- | --- | --- | --- |
| synchronous RPC | same thread | instrumented request ID | false positive join |
| workqueue helper | `work_struct` | explicit queue ID | worker 被算给错误 producer |
| async file server | `io_uring` | SQE/CQE operation ID | completion 与 submitter 断开 |
| Tokio service | Future 跨 worker | tracing span/task ID | 把 thread attribution 当 task attribution |
| Go service | 多 goroutine 完成一个 task | `runtime/trace` task | goroutine split 丢逻辑操作 |
| cache/buffer-pool workload | app-defined resource | explicit resource owner | kernel evidence 无法识别语义 |
| 大量短函数 workload | none | instrumentation | sampling coverage 系统性失真 |

主要指标应该是 causal-edge precision/recall、end-to-end request attribution accuracy、cost-attribution error、ID reuse 后的 false join，以及 aggregate estimator 的 confidence-interval coverage。overhead、memory、event volume、diagnosis latency 是约束，不是 correctness 的替代指标。

baseline 也必须够强：stack-only eBPF profiler、已有 instrumentation 时的 OpenTelemetry trace、Go/Rust runtime-native trace，以及 high-fidelity instrumented ground truth。如果新系统只比一个故意很弱的 stack-only baseline 好，论文结论并不够有说服力。

## 什么证据会改变这个结论？

bounded handoff graph 只有在 stack sampling 和 full tracing 中间确实存在一块有价值空间时才值得做。

有三类结果会明显削弱本文判断。

第一，如果现代 trace context 已经能低开销地覆盖几乎所有 CPU sample，并且自然跟随 kernel async work，那么额外做一层 eBPF causality 没有必要，OpenTelemetry profile-to-span link 就够了。

第二，如果 kernel-visible identifier 在真实对象复用下太不稳定，修好 join 又必须侵入 runtime，那么所谓 zero-instrumentation 优势就消失了。

第三，如果 exact handoff capture 加 state maintenance 的开销和一个工程良好的 trace 差不多，却提供更少 semantic detail，那么 causal question 应该交给 tracing，eBPF profiler 继续专注 stack。

相反，更有意思的结果是：少量 eBPF handoff probe 加可选 runtime join，就能用 continuous-profiling 级别的成本解释大部分 async CPU 与 wait attribution。那么“profile”和“trace”并不是唯二选择，中间还存在第三层：**用稀疏 causal evidence 解释 sampled work 为什么会运行，而不保存所有事件。**

这才是值得实现和认真测量的机制。

## 一手资料

- [OpenTelemetry eBPF Profiler](https://github.com/open-telemetry/opentelemetry-ebpf-profiler)
- [OpenTelemetry Profiles specification](https://opentelemetry.io/docs/specs/otel/profiles/)
- [Linux workqueue tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/workqueue.h)
- [Linux io_uring tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/io_uring.h)
- [liburing io_uring UAPI header](https://github.com/axboe/liburing/blob/master/src/include/liburing/io_uring.h)
- [Linux scheduler tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/sched.h)
- [Tokio task spawn documentation](https://docs.rs/tokio/latest/tokio/task/fn.spawn.html)
- [Rust tracing Instrument documentation](https://docs.rs/tracing/latest/tracing/trait.Instrument.html)
- [Go runtime/trace package](https://pkg.go.dev/runtime/trace)
- [OSDI 2026: Diagnosing Performance Issues in Application-Defined Resources](https://www.usenix.org/conference/osdi26/presentation/hu-yigong)
- [OSDI 2026: When Sampling Lies](https://www.usenix.org/conference/osdi26/presentation/devsot)
