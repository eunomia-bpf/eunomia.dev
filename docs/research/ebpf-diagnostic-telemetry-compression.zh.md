---
date: 2026-08-22
title: "eBPF 能压缩遥测数据而不丢失诊断依据吗？"
description: "持续运行的 eBPF 遥测会带来大量事件，而过度聚合又会删掉诊断所需的上下文。本文讨论如何用诊断契约、状态变化样本和覆盖度记录，在固定开销下压缩遥测，同时保留可验证的诊断能力。"
tags:
  - Daily Report
  - eBPF
  - Observability
  - Profiling
  - Telemetry
research_question: "持续运行的 eBPF 可观测系统，怎样在导出前减少遥测数据量，同时保留后续故障诊断真正需要的证据？"
source_cutoff: 2026-08-22
status: daily-report
---

# eBPF 能压缩遥测数据而不丢失诊断依据吗？

持续运行的 eBPF profiler 可以看到每一次系统调用、调度切换、内存分配、缺页、队列操作和应用层 probe。系统负载不高时，把这些事件都导出来看起来很自然；一旦进入生产规模，带宽、内存、存储和分析成本都会跟着事件速率增长。可如果太早把事件聚合成 counter 或 histogram，真正的故障一小时后出现时，调查者可能发现最需要的上下文已经被删掉了。

这就是 eBPF 遥测压缩真正困难的地方：**系统必须在不知道未来会问什么问题时，提前决定哪些信息可以丢。**

<!-- more -->

Linux 已经给了 eBPF 两类很适合在源头减少数据量的基础设施。[BPF map](https://docs.kernel.org/bpf/maps.html) 可以把状态留在事件发生附近，包含 array、hash、per-CPU、queue、stack、Bloom filter 等不同结构；[BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html) 可以高效传输可变长记录，并在一个共享 ring 中保留跨 CPU 的 reservation 顺序。一个 BPF 程序完全可以把数万次事件压成少量 map 条目，只把挑选出的记录发给用户态。

但这些原语没有回答一个更重要的问题：**压掉哪些信息以后，诊断还成立？**

一个 counter 可以告诉我们发生了 8,000 次等待，却未必还记得最长那次等待开始时，哪个 resource generation 正被谁持有。一个 histogram 可以保留延迟分布，却删掉让某个 tail request 变得异常的状态变化。一个 top-k map 可以留下最忙的 key，却可能把后来证明是故障依赖的那个罕见 key 淘汰掉。

本文把目标称为**保留诊断能力的语义压缩**。eBPF collector 不应该只追求 compression ratio，而应该明确声明压缩后的表示仍能回答哪些问题，为无法从 aggregate 重建的状态变化保留少量原始样本，并让每个 summary 携带自己的覆盖度和丢失信息。最终比较的核心指标不是“少了多少字节”，而是在相同采集预算下，调查者是否还能得到同样正确的 root cause。

这和之前的 [AI Agent 证据预算报告](https://eunomia.dev/zh/research/agent-trace-evidence-budget/) 不是同一个问题。那篇文章讨论整个 Agent observability 系统如何在代表性采样、incident、causal anchor 和 flight recorder 之间分配证据预算。这里把问题下沉到 collector：**高频系统遥测还没有离开 eBPF 数据路径之前，哪些内容可以变成 summary，哪些必须作为 exemplar 保留，下游又怎样知道哪些证据已经不存在？**

## eBPF 遥测压缩为什么不是普通压缩

让 observability 数据变小至少有三种不同方法，而它们保留的性质并不一样。

### 无损表示压缩保留原始记录的逻辑内容

[μSlope](https://www.usenix.org/conference/osdi24/presentation/wang-rui) 利用半结构化日志中的 schema 冗余做无损压缩。论文报告 21.9:1 到 186.8:1 的压缩比，最高达到 Zstandard 的 2.34 倍，并且可以不完全解压就执行搜索。[Tracezip](https://arxiv.org/abs/2502.06318) 对 distributed trace 使用相似思路：把重复的 span 结构放进 Span Retrieval Tree，backend 再利用已经收到的公共信息重建完整 trace。

这类系统的优势在于逻辑记录还在。某个字段虽然没有逐条原样传输，backend 仍然能够重建。

eBPF 在源头做 aggregation 则不同。如果 BPF 程序把 100,000 条 `(pid, resource, latency)` 事件替换成一张 histogram，未来就无法通过更好的 codec 恢复“哪一次事件属于哪个 resource generation”。这不是更高效地编码了信息，而是直接删掉了信息。

### 采样保留部分实例，不保证完整覆盖

[OpenTelemetry](https://opentelemetry.io/docs/concepts/sampling/) 区分 head sampling 和 tail sampling。head sampling 很早就做决定，成本低，但无法利用完整 trace 的信息挑选重要样本；tail sampling 可以看到大部分甚至全部 span 后再保留 error 或异常 trace，但采集链路仍需要先处理这些 span。

对于高频 eBPF 事件，越早做决定越能省掉真正的 hot-path 成本，也越容易造成不可逆的信息损失。一条 kernel-side event 一旦被过滤掉，后面的 tail sampler 没有办法再请求它缺失的字段。

采样还有一个 aggregation 未必具备的统计契约。如果每条事件都有已知的独立采样概率，可以给总体估计附带 uncertainty。一个手写 map 如果只留下“interesting” key，并在容量不足时静默淘汰其他 key，就没有这么简单的统计解释。

### workload-aware tracing 保留挑选过的执行结构

[StriaTrace](https://www.usenix.org/conference/osdi26/presentation/wu-haonan) 展示了在清楚 workload 结构时可以省掉多少 tracing 开销。它面向在线 LLM inference，重点追踪 synchronization point 和 critical path，只在异常时开启更详细的 tracing。论文报告相对替代方案降低 97.8% tracing overhead，并在实际使用中诊断了数百个异常，覆盖 19 类 root cause。

这个结果重要的地方不只是“少采一些事件”，而是系统知道哪些执行结构值得保留。StriaTrace 可以依靠 synchronization point 和 inference critical path，是因为它知道 serving stack 的结构。

通用 eBPF observability 层面对的程序和诊断问题更杂。它需要一种方式表达 workload-specific 的保留策略，同时又不能把某一个固定诊断写死进所有 probe。

## Linux 原语已经暴露了这个矛盾

BPF ring buffer 是 non-blocking 的。没有足够空间时，reservation 会失败；在 NMI context 中，即使 ring 没满，也可能因为内部 lock 无法获得而失败。`bpf_ringbuf_query()` 可以看到 available data 和 producer/consumer position，但内核文档明确提醒这些只是瞬时快照，更适合 debugging、reporting 或 heuristic，而不是稳定的事实。

因此，一个 exporter 可以既高效又不完整。如果 consumer 只看到已经 commit 的 record，就需要额外 accounting 才能区分“这里没有发生事件”和“producer 当时无法留下证据”。

BPF map 的问题不一样。它们非常适合保存 compact state，但结果的意义由 map 类型和更新程序共同决定。per-CPU map 可以减少部分跨 CPU 同步，LRU 变体则可能在容量达到上限时淘汰条目。某个 map value 对自己表示的状态可以完全精确，却仍然不足以回答未来出现的新问题。

所以缺少的不是另一个 map 类型，也不是另一个 compression codec，而是连接**诊断问题**和**被保留遥测**的契约。

## 用诊断契约决定源头应该保留什么

假设我们要用 eBPF 持续观察应用队列和 scheduler effect，原始事件可能包括：

```text
enqueue(queue_id, item_id, generation, ts)
dequeue(queue_id, item_id, generation, ts)
sched_in(pid, cpu, ts)
sched_out(pid, cpu, state, ts)
complete(item_id, status, ts)
```

这些记录无法无限保存。写 BPF 程序之前，collector 应该先声明后续至少要能回答哪些问题：

1. 哪些 queue generation 出现过长 residence time？
2. 一个 slow item 究竟是在 queue 里等、在 runnable 状态等 CPU，还是执行本身慢？
3. 有多少 item 没有被完整观察，或者采集过程中发生过丢失？
4. 某个罕见失败 item 是否还能恢复出足够的状态变化顺序？

这些就是**诊断义务**。它们可以直接决定 compact representation：

- 按 generation 保存 enqueue/dequeue/completion balance；
- 保存 queue residence 和 runnable delay 的 histogram；
- 用 bounded map 追踪仍 active 的 item generation；
- 对 first occurrence、状态切换、outlier 和 invariant violation 保留少量原始 exemplar；
- 记录 eligible event、成功更新、map eviction、ring-buffer reservation failure、probe/schema generation 等 coverage 信息。

最重要的是，这个表示要带一份机器可读的 answerability 说明。一个查询可能是 `exact`，也可能是“按照已知采样规则估计”、只能由 exemplar 支撑，或者直接 `unavailable`。Dashboard 不应该把这四种情况都显示成同样确定的数字。

## 现有研究还缺什么

### 1. 压缩系统主要优化字节，而 profiler 真正需要优化的是问题还能不能回答

μSlope 和 Tracezip 这样的无损系统可以直接比较 compression ratio，因为逻辑输入仍然可以重建。源头 aggregation 的目标不同：真正重要的是哪些 diagnosis 在 reduction 以后仍然成立。

很多 eBPF collector 现在把这个决定隐含在代码里。一个程序存 histogram，另一个按 PID 存 counter，第三个只在超过 threshold 时发 record。实现可能很高效，却没有一个公共 artifact 说明这些 reduction 之后哪些查询仍然有效。

缺少的是从 diagnostic obligation 到 retained state 的明确映射。最直接的实验，是准备一组 root cause 已知的 incident replay，在相同 CPU、memory 和 export budget 下，对比 raw trace 与不同 compact representation 最终得到的 diagnosis。

### 2. 丢失信息经常和被它影响的 summary 分开记录

ring-buffer reservation failure、map eviction、probe 不可用、schema mismatch、collector restart 都会改变一个 summary 的含义。但现实 telemetry pipeline 经常只输出最终 counter 或 histogram，不把这些 coverage condition 附在结果上。

例如，一个 per-resource latency histogram 覆盖了 95% 事件，但缺掉的 5% 恰好都发生在 ring buffer 饱和的时候。把这张 histogram 当作无偏样本，可能比直接说“现在证据不足”更危险。

缺少的机制是**携带覆盖度的遥测结果**：每个 compact result 都要标识 collection generation，并带上和自己相关的 evidence-loss counter。可以通过主动制造 ring-buffer pressure、map eviction、probe removal 和 process restart，测量报告的 confidence 是否会随着真实误差一起下降。

### 3. trigger 无法恢复已经被 summary 删掉的历史上下文

只在异常时做 detailed tracing 很有吸引力，因为正常执行通常占据绝大多数时间。StriaTrace 也证明了 workload-aware escalation 可以非常有效。

边界出现在 trigger 晚于关键状态变化时。一次 5 秒 latency anomaly 可能来自 30 秒前的一次 queue ownership change。tail request 变慢以后再打开 raw tracing，并不能把旧 ownership event 找回来。

缺少的是和 semantic state 绑定的、小而有界的**pre-trigger exemplar history**，而不是只按时间保存所有 raw event。它应该让最可能解释未来状态变化的 transition 留得更久，同时允许重复的 steady-state event 被 summary 吸收。

### 4. 手写 aggregation 无法说明什么时候应该换一种表示

BPF map 很容易构建一套高效 summary，却不会自动告诉 collector：当前 key cardinality 已经暴涨、top-k 集合变得很不稳定，或者 invariant failure 多到 raw exemplar 比继续更新 counter 更有价值。

这正好连接到这个系列下一步的 adaptive collection 问题。但 adaptive collector 在安全改变 fidelity 之前，需要先有一个稳定契约说明每种 representation 保留了什么。否则所谓“自适应可观测性”只是一些 heuristic，系统没有办法判断切换以后是否还支持原来的 diagnosis。

## 兼具学术价值与生产价值的方向

### 1. 把诊断义务编译成 eBPF retention plan

**缺口。** 现在 operator 往往手工选择 BPF map、filter、threshold 和 export field。程序能跑，却没有机器可读的说明告诉下游这些 summary 可以支撑哪些 diagnosis。

**机制。** 定义一个小型 diagnostic contract，描述必须保留的 entity、transition、join、accuracy bound，以及允许采用的 approximation。Compiler 再把它映射成 retention plan：

```yaml
question: queue_delay_attribution
entities:
  - queue_generation
  - item_generation
required_transitions:
  - enqueue
  - dequeue
  - sched_in
  - complete
outputs:
  queue_latency:
    mode: histogram
    exact_count: true
  slow_item_examples:
    mode: bounded_exemplars
    retain: first,last,outlier,invariant_violation
coverage:
  track:
    - eligible_events
    - map_evictions
    - ringbuf_reservation_failures
```

Compiler 选择 BPF map layout、per-CPU 或 shared state、需要 export 的 record，以及 userspace reconstruction 逻辑；同时生成 query manifest，说明每个结果是 exact、estimated、由 exemplar 支撑，还是不可回答。

**和现有工作的区别。** Tracezip 与 μSlope 在保留 record 可重建性的前提下做 compression。已有 eBPF 工具会在源头 aggregate，却把 diagnostic contract 留在手写代码里。这里要编译的性质是“reduction 以后还能够回答什么”。

**Artifact。** 一个小型 compiler、可复用的 libbpf/BPF template、query manifest，以及同时回放 raw 与 compact trace 的 replay harness。

**评测。** 使用 scheduler、network、memory 和 application-resource incident，并提供 ground-truth root cause。让 raw export、手写 eBPF aggregation、概率采样和 compiled plan 在相同 CPU、map memory 与 byte budget 下运行。比较 root-cause accuracy、false attribution、query coverage、export bytes、BPF runtime cost 和 map pressure；再去掉 compiler，使用人工选择的 aggregate 做 ablation。

**学术价值。** 这里研究的是 observability reduction 能否保留一组声明过的诊断性质，而不是单纯最小化数据量。

**生产价值。** 团队可以先说清 always-on monitor 必须回答什么，再生成一个有明确预算的 collector，不需要每张 dashboard 都维护一套独立的 ad-hoc BPF 程序。

**失败条件。** 如果真实 diagnosis 无法用较小 contract 表达，必须把大部分应用逻辑都塞进 schema，那么 hand-written collector 更简单，这个抽象就没有成立。

### 2. 在 compact summary 旁边保留状态变化 exemplar

**缺口。** aggregate 很适合描述 steady state，却容易删掉解释罕见 transition 的顺序；triggered tracing 又可能启动得太晚。

**机制。** eBPF data path 同时保留两层证据。第一层是常规 compact map state，包括 counter、histogram、active generation 和低基数 summary。第二层是按 semantic entity 或 generation 索引的 bounded exemplar cache。

只有真正增加 transition 信息时才留下 exemplar，例如某一 generation 的第一次事件、retire 前最后一次事件、invariant violation、类别变化、threshold crossing 或罕见 outlier。重复 steady-state event 只更新 aggregate，不再分配一条 raw record。当 userspace 请求 escalation 或检测到 incident 时，collector 导出这些 exemplar 和当前 summary。

这个 cache 可以用 bounded map 保存 entity-local state，用 ring buffer 提交最终 capsule。Eviction policy 必须显式可见，避免把“exemplar 不存在”误解成“transition 没发生”。

**和现有工作的区别。** 普通 flight recorder 保存最近一段时间的 raw event；这里保留的是稀疏的**状态变化历史**。因此一个更早但语义上关键的 transition 可以活得更久，而大量更新但重复的新事件会被丢掉。

**Artifact。** 一组适合常见 eBPF observability 模式的 exemplar library，以及把 summary 和 transition 合并成 incident capsule 的 userspace 格式。

**评测。** 构造关键 transition 在可见症状前 1 秒、10 秒和 60 秒发生的 incident。固定 memory/export budget，对比 full raw export、固定大小 time ring、tail-triggered tracing 与 transition exemplar。测量真实 transition 的保留率、root-cause accuracy、false explanation 和 overhead。

**学术价值。** 这个实验直接测试 semantic change 是否比时间新旧更适合作为在线系统诊断的保留单位。

**生产价值。** Always-on monitor 可以在不连续保存 raw event stream 的情况下，留下足够的 pre-incident context 解释低频故障。

**失败条件。** 如果简单 time-based ring 在多种 workload 下用相同 memory 就能保留同样的诊断上下文，那么 semantic exemplar selection 没有必要。

### 3. 让每个压缩结果自己携带证据覆盖度

**缺口。** compact value 经常看起来非常精确，即使输入证据其实不完整。

**机制。** 为每个 summary 增加 collection generation 和 compact coverage record。不同 retention plan 可以记录：

- eligible event 数量；
- 真正进入 summary 的 event 数量；
- 已知概率采样率；
- ring-buffer reservation failure；
- 和当前 key space 有关的 map insert failure 或 eviction；
- probe/schema generation；
- collector restart epoch；
- invariant violation 或 unknown entity identity 数量。

Userspace 在展示 value 之前先和 coverage record join。这样 diagnosis 可以写成“queue residence 上升，当前 event coverage 为 99.8%”，也可以直接写“active-item map 淘汰了 18% generation，因此 attribution 不可用”，而不是给出一个看起来精确、实际却隐藏 loss 的数字。

第一版应该刻意保持 non-adaptive。它先负责告诉下游“这个 representation 什么时候不再可信”。之后的 controller 才有依据决定何时提高 fidelity，以及提高到什么程度。

**和现有工作的区别。** 采样系统记录 probability，aggregate estimator 因而可以计算 uncertainty。eBPF monitoring 的 coverage 更复杂，因为证据损失还可能来自 buffer pressure、bounded state、missing probe、restart 和 semantic mismatch，而不是单一 sampling probability。

**Artifact。** 一个公共 coverage schema、一组 BPF-side accounting helper，以及能把 coverage 传播到 query 和 alert 的 downstream library。

**评测。** 通过 ring-buffer saturation、LRU eviction、关闭 probe、collector restart 和 schema change 主动注入不同类型的 loss。比较 reported coverage 与真实 query error 的 calibration，并测试 coverage-aware diagnosis 是否比完全相同但没有 coverage metadata 的 collector 更少给出错误 root cause。

**学术价值。** 它把 observability completeness 变成 compact representation 自己携带、可以测量的系统性质。

**生产价值。** Operator 能区分“metric 正常”和“collector 没有足够证据下结论”。对于罕见 incident，这个区别往往比多一位小数更重要。

**失败条件。** 如果 coverage metadata 不能预测 diagnosis error，或者 downstream 已经可以可靠推断同样的信息，那么 hot path 上增加这些 accounting 不值得。

## Benchmark 应该测诊断保留率，而不是只测 compression ratio

一个有意义的 benchmark 需要 full raw ground truth，并给所有方案相同资源预算。Workload 还要故意包含适合不同 representation 的情况。

每个 incident 都在预算路径之外保存一份完整 reference trace。候选 collector 则限制 BPF CPU time、map memory、每秒 export byte 和 userspace processing budget。随后执行需要不同证据的 diagnosis query：

| Incident | 必须保留的证据 | 容易生成的 summary | 容易丢失的信息 |
| --- | --- | --- | --- |
| queue buildup | enqueue/dequeue generation | residence histogram | 罕见 ownership transition |
| scheduler delay | runnable/scheduled interval | per-process delay total | 哪个 queue item 被阻塞 |
| memory regression | allocation/page generation | bytes per stack | COW/reclaim transition |
| network retry storm | connection/request generation | retry counter | 第一个失败 dependency |
| stale semantic probe | probe generation 与 invariant | event rate | 语义已经改变的证据 |

Benchmark 至少要报告：

- root-cause accuracy 与 false attribution；
- 仍然可以回答的 diagnosis query 比例；
- stated coverage 和实际 error 的 calibration；
- raw-event bytes 和 exported bytes；
- BPF execution overhead 与 map memory；
- 找到足够诊断证据所需的时间。

一个 collector 即使做到 1000:1 compression，只要它删掉了区分 queue delay 和 scheduler delay 的那条信息，就不比能保留关键 transition 的 20:1 方案更好。反过来，如果复杂 exemplar 对 diagnosis 没有比一张 histogram 带来任何提升，系统只是多花了复杂度。

## 哪些结果会改变这个判断？

本文依赖两个假设。第一，完整导出 raw eBPF event 的成本确实高到会限制 always-on deployment；第二，未来 diagnosis 虽然不能完全预测，但存在一组稳定核心问题，可以提前声明。

有两类结果会削弱这两个假设。

第一，如果无损方案可以用和 semantic aggregation 接近的 CPU、memory 和 bandwidth 成本编码并导出完整相关事件，那么保留 raw evidence 更简单。Tracezip 与 μSlope 已经说明这必须是认真对比的 baseline，而不是假想敌。

第二，如果真实 incident 经常需要 diagnostic contract 无法提前预测的字段和关系，那么源头 semantic reduction 可能过于脆弱。更稳妥的设计会保留更大比例、采样概率已知的 raw sample，或者更连续的 flight-recorder state。

真正有区分度的实验，是在多个系统领域做 equal-budget incident replay。如果 hand-written aggregate、概率采样或无损压缩在 root-cause accuracy 和 query coverage 上都能达到同样结果，就没有必要再增加 compiler 和 exemplar layer。

如果 diagnostic contract 可以稳定用更低 export cost 保住同样的 diagnosis，那么 eBPF observability 的目标就不应该只是“少发一些事件”，而应该是：**删掉重复信息，同时让被删掉的部分本身仍然可见、可判断。**

## 参考资料

- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html)
- [OpenTelemetry: Sampling](https://opentelemetry.io/docs/concepts/sampling/)
- [Tracezip: Efficient Distributed Tracing via Trace Compression](https://arxiv.org/abs/2502.06318)
- [StriaTrace: Efficient Tracing and Diagnosis for Online LLM Inference](https://www.usenix.org/conference/osdi26/presentation/wu-haonan)
- [μSlope: High Compression and Fast Search on Semi-Structured Logs](https://www.usenix.org/conference/osdi24/presentation/wang-rui)
- [eBPF 能理解应用自己定义的资源吗？](https://eunomia.dev/zh/research/ebpf-application-resource-semantics/)
- [异步系统里，eBPF Profiler 还要追踪什么？](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/)
- [性能分析器的采样什么时候会产生偏差？](https://eunomia.dev/zh/research/profiler-sampling-bias/)
- [AI Agent 轨迹到底该保留什么：固定证据预算下的可观测性设计](https://eunomia.dev/zh/research/agent-trace-evidence-budget/)
