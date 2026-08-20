---
date: 2026-08-20
title: "性能分析器应该如何发现应用自定义资源？"
description: "应用内部的 buffer pool、cache 和 queue 可能不会表现为系统资源压力。本文分析 profiler 如何发现这些资源的语义，并在软件升级后识别过期的资源模型。"
tags:
  - Daily Report
  - Profiling
  - Observability
  - Performance
  - Linux
research_question: "Profiler 如何发现 application-defined resource 的身份、生命周期、容量与使用语义，在运行时验证这些语义，并在推断或声明的资源模型过期时明确发现它？"
source_cutoff: 2026-08-20
status: daily-report
---

# 性能分析器应该如何发现应用自定义资源？

假设一个数据库的吞吐突然下降，但 CPU 利用率、常驻内存和锁竞争都没有明显异常。真正的问题可能是某个请求把内部 buffer pool 填满了临时数据，后续请求不得不反复淘汰页面并重新读取磁盘。操作系统看到的仍然只是数据库很早以前就分配好的那块内存，因此普通系统指标可以完全正常。

这里真正影响性能的资源不是内核里的 page allocator 或某把锁，而是应用自己定义的 buffer pool。它的容量、所有权、淘汰策略和生命周期都存在于应用逻辑里。

这就是 **application-defined resource profiling** 面临的可见性缺口。近期研究已经证明，Profiler 可以从源码语义、静态分析和运行时行为里恢复大量隐藏的资源结构。下一步更难的问题不是再多找到几个函数，而是保证恢复出来的资源模型在软件持续变化以后仍然正确。Profiler 需要知道一个资源实例怎样标识、这个标识什么时候会被复用、获取和释放的单位是什么、哪些操作代表压力，以及当前模型是否仍然匹配正在运行的 binary。

<!-- more -->

因此，更值得做的系统机制是一个**带版本的资源语义契约**。应用可以显式声明其中一部分，离线分析可以补全缺失部分，运行时观测再负责验证。如果真实行为和契约开始不一致，Profiler 应该降低置信度，而不是继续输出一张看起来很完整的错误 profile。

Linux `user_events`、静态 trace marker、uprobe、编译器插桩和 eBPF 都可以提供证据，但它们本身都没有定义“这个应用资源到底是什么”。所以这篇报告讨论的是通用 profiling 问题，而不是把某一种采集机制包装成答案。

这和之前的[异步 eBPF causal profiler](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/)问题相邻，但边界不同。那篇报告问的是：逻辑工作离开原线程以后，怎样把不同 execution context 重新接起来。这里的问题更早一步：**Profiler 最开始怎样获得一个可信的应用资源身份和生命周期？**

## 为什么系统指标可以完全正确，却仍然看不到瓶颈

OSDI 2026 论文 *Diagnosing Performance Issues in Application-Defined Resources* 给出了一个很具体的 MySQL 例子。临时表页面可以占满 InnoDB buffer pool，后续请求随后承担昂贵的页面淘汰和重新读取成本。由于 buffer pool 本来就是 MySQL 预先分配并自行管理的内存，操作系统未必会看到 memory pressure。CPU、memory、lock 和 I/O profiler 可以看到症状，却不知道真正需要分析的资源抽象是什么。

论文分析了 45 个真实性能问题，在数据库和搜索系统中识别出 38 种不同的应用自定义资源，包括类似内存池的资源、日志和索引等共享状态，以及内部任务队列。论文提出的分析模型能够覆盖其中 39 个案例。这里更重要的不是具体分类，而是一个事实：**应用逻辑会创造自己的容量和所有权规则，而这些规则不一定对应任何内核对象。**

[gigiprofiler](https://homes.cs.washington.edu/~baris/public/gigiprofiler.pdf) 使用一条混合流水线来恢复这些语义。LLM 根据名字、注释、文档和周围代码提出候选资源和使用位置；静态分析再根据 control flow 与 data flow 验证这些候选；运行时随后插桩四类交互：`WAIT`、`ACQUIRE`、`USE` 和 `RELEASE`，并把它们与请求关联起来。

这几层验证很重要，因为纯语义推断和纯程序分析都不够可靠。在论文的 MySQL 实验里，只使用 LLM 时，不同模型产生的 false positive rate 在 45% 到 60% 之间。静态验证会继续减少 false positive，运行时验证还能进一步过滤候选。另一方面，对 MySQL buffer pool 的穷举分析显示，不同模型平均仍会漏掉约 23% 的 resource-use event。

尽管资源图并不完整，gigiprofiler 仍然成功诊断了实验中复现的 15 个问题，并另外发现了两个后来得到开发者确认的 MariaDB 问题。这说明一件很有意思的事：**资源发现不需要完美才能有诊断价值，但“能诊断”不等于“资源语义已经稳定”。**

## 找到一个 event site，还没有得到一个资源模型

假设 Profiler 已经正确找到下面几个调用点：

```text
get_page(pool, key)
release_page(pool, page)
evict(pool)
```

仍然有几类关键语义没有确定。

第一是**身份**。`pool` 是整个进程唯一的对象、每个 tenant 一个、每个 shard 一个，还是一个可能切换 backing object 的 wrapper？`page` 本身是独立资源，还是从 `pool` 借出的一份容量？

第二是**生命周期**。一个地址释放后可能再次被 allocator 使用。长时间运行的 Profiler 如果把 pointer 当作永久唯一 ID，就可能把两个完全不同的资源实例拼到一起。

第三是**单位和容量**。`ACQUIRE(pool, 1)` 只有在“1”有稳定含义时才有价值。一个 buffer page、一个 connection、一个 queue credit、一个 token slot 和一个 byte 代表的容量完全不同。有些资源存在硬上限，有些资源会持续增长，直到某个策略阈值触发回收。

第四是**状态变化**。名字叫 `get` 的函数可能是在真正占用容量，也可能只是查询已有对象，或者返回一个 borrowed reference。`release` 也未必意味着容量马上可以再次使用，它可能只是把对象放进延迟回收队列。函数名只能提供线索，不能直接等同于语义。

最后是**版本**。软件升级以后，一个 wrapper 可能被 inline，一个 pool 可能被拆成多个 shard，allocator 可能更换，counter 的意义也可能改变。高层功能看起来没有变化，但昨天生成的 resource model 已经不一定适用于今天的 binary。

这些错误会直接污染 attribution。只要 Profiler 错误地连接了两个 object generation，或者把借用引用当作资源所有权，一张视觉上很合理的 resource flamegraph 也可能在语义上完全错误。

## Linux 已经提供了多种证据入口

资源语义契约不需要绑定一种插桩技术。

Linux 的 [`user_events`](https://docs.kernel.org/6.18/trace/user_events.html) 允许用户进程注册带类型的 trace event，再交给 ftrace、perf 等现有工具消费。应用还可以知道当前 event 有没有被启用，没有消费者时就不必发送数据。当前 Linux 文档还定义了 multi-format registration，同一个逻辑 event name 可以同时存在多个 payload format。这已经解决了一个很重要的传输和 schema 演进问题。

但是一个 event schema 仍然不会告诉 Profiler：`pool_id` 只在资源销毁前唯一，或者 `units` 表示 page 而不是 byte。这里缺失的是比 payload 格式更高一层的资源语义。

Linux [uprobe](https://docs.kernel.org/6.18/trace/uprobetracer.html) 提供了另一种取舍。Profiler 无需修改应用，就可以在现有用户态代码位置插入 probe，读取参数或返回值，也可以统计命中次数。这很适合给已有 binary 补 observability。问题是 attach point 仍然只是某个 executable 或 library 中的代码位置，Tracer 还需要知道哪个值才代表真正的逻辑资源。

静态用户态 marker 和编译器插桩则能表达更稳定的开发者意图。gigiprofiler 走的是另一条路：自动推断 event，再通过 LLVM pass 插入探针。这些方式并不冲突。真正缺的是一个中间表示，让不同 Profiler 可以比较、复用并验证它们对资源语义的判断。

## Profiler 应该携带一个带版本的资源语义契约

这个契约不需要成为庞大的通用 ontology。它只要把 profiling 真正依赖的假设写清楚：一个 event 影响哪个资源实例，这个身份在多长时间内有效，变化了多少资源量，以及当前映射有多可信。

概念上，一个资源类别可以表示成：

```text
ResourceClass {
    name: "buffer_pool"
    schema_version: 3
    build_identity: <binary or module identity>
    instance_key: <expression or declared field>
    generation_rule: <creation/destruction boundary>
    unit: "page"
    capacity: <fixed, dynamic, or unknown>
    events: {
        acquire: ...
        use: ...
        wait: ...
        release: ...
    }
    scope: <process, tenant, shard, request, ...>
    evidence: <declared, statically-validated, inferred>
}
```

契约可以来自三种证据。

1. **显式声明。** 应用或 library 在 `user_events`、USDT 风格 marker 或其他稳定 instrumentation API 旁边给出一个很小的 descriptor。
2. **自动推断。** 当没有 descriptor 时，源码或 binary 分析生成候选资源类别和 event mapping。
3. **运行时不变量。** Profiler 检查真实执行是否继续符合这些语义，例如 generation 是否发生重叠、容量账本是否变得不可能成立、原本应该存在的 release 路径是否消失。

第三类证据决定了这个系统能不能长期工作。没有运行时验证，自动生成的契约很容易变成另一种过期配置文件。

## 现有研究还缺什么

### 资源发现主要测 event site 对不对，却很少单独测资源身份对不对

gigiprofiler 对 false positive 和漏掉的 resource-use event 做了很细的评估，这是必要的。但 Profiler 即使找到了正确函数，也可能把函数参数解释成错误的逻辑 identity，或者不知道这个 identity 在什么时候失效。

缺少的实验是把 **event-site correctness** 和 **resource-instance correctness** 分开测。Benchmark 应该故意复用地址、把 pool 做成 shard、传递 borrowed reference，并在不同版本之间修改 ownership rule。除了 event precision/recall，还要测 false join 和 identity split。

生产环境里，这类错误尤其危险，因为一次 false join 可能把一个 tenant、request 或 generation 的成本归到另一个对象上。即使绝大多数 probe site 都找对了，高 fan-out 资源上的少量语义错误也可能让最终判断失真。

### 显式 trace schema 能描述 payload，却没有统一描述生命周期

`user_events` 可以定义字段类型，也能让同名 event 的多个格式共存；静态 marker 也可以提供稳定的命名位置。但这些接口都没有一个公共语言来表示 generation、容量单位、所有权、借用和延迟回收。

缺少的不是新的 event transport，而是一层很小的 semantic descriptor，并且这个 descriptor 必须绑定到具体 build 或 module version。

一个直接的验证标准是跨工具复用：如果 perf、一个 eBPF Profiler 和应用专用 debugger 读取同一个 descriptor 后仍无法对资源 identity 达成一致，那么语义还没有真正从某个 Profiler 实现中独立出来。

### 自动推断缺少足够强的“模型已经过期”信号

OSDI 2026 的结果已经说明为什么需要多层验证。纯 LLM discovery 的 false positive 很高，静态和运行时分析可以继续过滤；论文也观察到注释缺失或 validation rule 不匹配时会漏掉真正的 event。

软件演进会把同一个问题放大。版本 N 推断出的 mapping 在版本 N+1 可能只是部分失效。如果 Profiler 没有明确的负面信号，它就会继续生成很干净的 profile，却不知道自己的资源模型已经过期。

可以检查的信号包括不可能成立的容量平衡、重叠的 object generation、event 大量迁移到从未见过的 call site，或者 binary build identity 变化。这里最重要的性质不是自动修复一切，而是**明确降低置信度**。

### 不同插桩策略之间缺少统一的资源语义 benchmark

手写 annotation、静态 marker、编译器插桩、自动推断、uprobe 和混合方案付出的工程成本与运行时成本不同，目前很难在一套 ground truth 上公平比较。

需要一个 benchmark 明确知道每个资源的 identity、lifetime、capacity 和 ownership，并提供两类软件变化：一种只改变实现形状，不改变资源语义；另一种真正改变语义，必须让旧 descriptor 失效。

没有这套 benchmark，仅仅报告低 runtime overhead 并不能回答维护成本和 false confidence 风险。

## 兼具学术价值与生产价值的方向

### 1. 可移植的资源语义 manifest

**缺口。** Event format 可以携带值，自动 Profiler 也能找到 resource-use site，但两者都没有产生一个小而可复用的资源身份和生命周期描述。

**机制。** 定义一个带版本的 manifest，描述 resource class、instance key、generation boundary、计量单位、capacity semantics、ownership scope 和 event mapping，并绑定 binary build ID 或 module identity。显式 descriptor 和自动推断都输出同一种结构，每个字段同时标记证据来源和置信状态。

**与现有工作的差别。** 相比 gigiprofiler，重点不是再实现一个 detector，而是把 detector 的结果变成其他工具可以复用、并能跨升级检查的 durable artifact。相比 `user_events`，manifest 说明的是资源含义，而不是 event payload 的字节布局。

**产物。** 一个开放 schema，为若干代表性应用实现 compiler/runtime adapter，再给 perf 或 pprof 风格工具实现 reader。

**评测。** 选择数据库、Web server、runtime 和 model-serving cache，并人工建立 ground truth。比较手工插桩、`user_events` 或静态 marker、gigiprofiler 风格推断，以及 manifest pipeline。指标包括 descriptor 编写成本、event precision/recall、resource-instance precision/recall、false join、attribution error、runtime overhead 和跨版本复用率。

**学术价值。** 核心问题是 semantic observability 能否成为独立于某一种 tracing mechanism 的可移植契约。

**生产价值。** 运维团队可以稳定地分析内部 pool、queue、cache 和 credit，同时自由替换底层 collector。

**失败条件。** 如果每个应用维护 descriptor 的成本接近手写诊断逻辑，或者纯自动推断在版本升级后仍然同样准确，那么 manifest 没有带来足够收益。

### 2. 运行时验证契约，并允许置信度显式下降

**缺口。** Descriptor 即使语法合法、probe point 仍然存在，也可能已经表达错误的语义。

**机制。** 运行轻量 validator，检查 semantic invariant 而不只检查 event 有没有送到。可以验证 generation 是否唯一、acquire/release 的平衡范围、已知 capacity 的边界、合法状态顺序，以及高层操作是否覆盖预期 event。当证据和契约冲突时，把对应 resource class 标记成 degraded 或 unknown，而不是继续输出普通 attribution。

具体 observer 可以按应用选择：显式 event、compiler hook、uprobe、eBPF 或组合都可以。昂贵检查只在低成本 invariant 失败后再开启。

**与现有工作的差别。** gigiprofiler 的 post-profiling validation 会根据运行时 workload 过滤候选 false positive。这里把 validation 变成持续的生产属性，并且把“语义可能过期”传递给下游消费者。

**产物。** 一个 validation runtime、写进 profile record 的 confidence model，以及一套在不更新 descriptor 的情况下修改资源实现的 fault-injection tests。

**评测。** 测量 stale contract 的发现延迟、false alarm、阻止了多少错误 attribution、运行时开销以及 descriptor 更新后的恢复速度。测试既包括只改函数名或代码结构、语义不变的版本，也包括真正改变 ownership 或 lifecycle 的版本。

**学术价值。** 这是一个可观测性系统怎样判断“自己的语义模型已经不可信”的问题。

**生产价值。** 过期的 Profiler 集成会变成显式健康问题，而不是悄悄制造错误性能结论。

**失败条件。** 如果 invariant 检查既抓不到大部分语义漂移，又经常把正常 workload 变化误报成错误，那么它不能作为可信度信号。

### 3. 面向 application-resource profiling 的 ground-truth benchmark

**缺口。** 现有 Profiler 评测通常知道选定 bug 的 root cause，却不知道整个执行过程中每个资源实例和状态变化的完整真值。

**机制。** 构建一个 workload harness，在运行 workload 的同时生成资源 ground truth。覆盖固定 pool、弹性 cache、有界 queue、可复用 object slot、borrowed reference、延迟回收、sharded ownership 和跨请求干扰。再加入受控的软件变换，例如 inline、增加 wrapper、更换 allocator、复用 object address 和改变 schema version。

**与现有工作的差别。** 这套 benchmark 测的是普通 CPU profile 和 bug reproduction suite 没有覆盖的 semantic layer。它还能区分“找到几个有用 hot site”和“正确重建资源生命周期”这两个能力。

**产物。** 开放 workload、truth trace、成对的软件 mutation/version，以及针对 event 与 resource semantics 的评分工具。

**评测。** 在相同 overhead budget 下比较纯系统 profiling、手写 annotation、静态 user event、动态 uprobe、自动推断和混合方案。报告 event-site accuracy、identity/lifetime accuracy、root-cause ranking、attribution error、false confidence、维护成本和运行时开销。

**学术价值。** 它把 semantic observability 变成一个可复现的 measurement problem，而不是继续积累不同 Profiler 的案例。

**生产价值。** 工具开发者可以在把新 collector 或 inference model 放进生产环境前，先验证它是否保持了资源语义。

**失败条件。** 如果即使主动破坏 identity 和 lifecycle reconstruction，最终性能诊断仍然几乎不受影响，那么额外的语义机制并没有改变目标决策。

## 哪些结果会改变这个判断？

这里的判断建立在两个假设上：应用自定义资源足够经常地影响真实性能，而且软件演进足够快，使过期的语义 mapping 成为实际问题。OSDI 2026 的证据已经说明隐藏资源可以导致严重性能问题，也说明自动发现能够诊断这些问题，但它没有证明每个生产 Profiler 都需要长期维护一个 resource contract。

如果自动推断可以跨大幅软件升级仍然保持稳定准确，或者应用已经通过现有 trace event 暴露了足够完整的生命周期语义，又或者运维只需要针对固定源码版本做一次性诊断，那么更简单的方案应该胜出。此时再加 manifest 和在线 validator 只会增加维护成本。

最有区分力的实验应该是 longitudinal study：选择几个持续演进的大型应用，把 Profiler 知识冻结在版本 N，然后测后续 commit 中 attribution 与 diagnosis 的准确度怎样下降。比较完全不适配、每次重新 discovery、显式 descriptor，以及 descriptor 加运行时验证四种策略。如果 stale model 很少导致错误决策，这层契约没有必要；如果错误会快速积累，那么 Profiler 就应该把“语义模型本身是否健康”也纳入 observability。

## 参考资料

- Yigong Hu 等，[*Diagnosing Performance Issues in Application-Defined Resources*](https://homes.cs.washington.edu/~baris/public/gigiprofiler.pdf)，OSDI 2026。
- Linux kernel documentation，[`user_events`: User-based Event Tracing](https://docs.kernel.org/6.18/trace/user_events.html)。
- Linux kernel documentation，[Uprobe-based Event Tracing](https://docs.kernel.org/6.18/trace/uprobetracer.html)。
- SystemTap documentation，[Static user-space probe points](https://sourceware.org/systemtap/langrefse4.html#x33-330004.5.7)。
- Eunomia 每日报告，[异步系统里，eBPF Profiler 还要追踪什么？](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/)。
- Eunomia 每日报告，[性能分析器的采样什么时候会产生偏差？](https://eunomia.dev/zh/research/profiler-sampling-bias/)。
