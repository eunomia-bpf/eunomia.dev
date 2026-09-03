---
date: 2026-09-03
title: "eBPF Profiler 什么时候该停止相信自己的诊断？"
description: "eBPF Profiler 即使丢事件、漏探针或语义过期，仍可能输出看似完整的诊断。本文讨论如何跟踪证据缺口、按需升级采集，并在证据不足时明确拒绝下结论。"
tags:
  - Daily Report
  - eBPF
  - Observability
  - Profiling
  - Reliability
research_question: "当事件丢失、探针缺失或语义漂移使证据不足时，eBPF Profiler 应该怎样识别诊断已经不再成立，并决定升级采集还是拒绝下结论？"
source_cutoff: 2026-09-03
status: daily-report
---

# eBPF Profiler 什么时候该停止相信自己的诊断？

设想一个长期运行的 eBPF Profiler 正在分析一次线上慢请求。它还能看到 scheduler delay、socket activity、部分应用探针，也能把这些事件串成一条像样的 root cause：请求主要卡在 run queue。

问题是，故障发生时 ring buffer 恰好被打满，一个协议 parser 只能看到固定长度的 payload，软件升级后还有一个可选 probe 没有继续覆盖原来的代码路径。Profiler 没有停止工作，也没有完全失去数据，它仍然可以输出一个完整的解释。真正难判断的是：**这个解释现在还有足够证据支持吗？**

<!-- more -->

“有事件丢失”与“当前诊断已经不可信”不是同一件事。即使 collector 很准确地知道丢了 2% 的记录，也不知道这 2% 是均匀分布的普通事件，还是刚好包含区分 CPU contention、锁等待和 I/O 等待的那一次状态变化。

2026 年 8 月的一组 [OpenTelemetry eBPF Instrumentation](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation) issue 把这个问题表现得很直接。[Issue #3067](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/3067) 记录了这样一种情况：请求实际上返回 200，但 instrumentation 没有在 `tcp_close` 前观察到 response，于是最终生成带合成 HTTP 499 的 client span。[Issue #2958](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2958) 则说明当前 BPF ABI 与 userspace decoder 的 16 KiB 上限会直接丢掉更大的合法 Go Auto SDK span。更早的 [#1381](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/1381) 和 [#2174](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2174) 还分别暴露了 `traceparent` 超出捕获窗口导致 trace context 断裂、请求大于配置 buffer 后 GenAI trace 无法正常生成的问题。

这些 issue 来自同一个项目，因此不能当成四份独立研究证据。但它们共同说明：eBPF observer 完全可能“还活着、还有数据、还能输出结果”，同时已经因为 transport loss、固定捕获窗口、parser 上限或遗漏路径失去决定性证据。

Linux 本身能够暴露一部分底层损失。[BPF ring buffer 文档](https://docs.kernel.org/bpf/ringbuf.html) 明确说明，空间不足时 reservation 会失败；在 NMI context 中，即使 ring buffer 没满，也可能因为拿不到 reservation lock 而失败。`bpf_ringbuf_query()` 可以提供 producer/consumer 等瞬时状态，但内核文档把这些值定位为 debugging、reporting 和 heuristic 信息，而不是稳定事实。对于 perf buffer，[libbpf](https://libbpf.readthedocs.io/en/latest/api.html) 还提供 `lost_cb`，让 userspace 知道发生过 record loss。

这些接口解决的是局部问题：某个 transport 有没有丢记录。它们没有直接回答另一个问题：**我现在准备给出的这条诊断，是否还被现有证据支持？**

本文的判断是，eBPF Profiler 需要在“采到了什么”与“允许得出什么结论”之间增加一个 **diagnosis validity contract**。Runtime 不只统计全局 loss rate，而是按具体诊断需要的证据义务跟踪缺口。当某个必要信号失效时，要么在固定资源预算内针对性补采，要么把结论降级成 `unknown`，而不是让剩下的数据继续制造精确感。

这与之前的 [eBPF 遥测压缩报告](https://eunomia.dev/zh/research/ebpf-diagnostic-telemetry-compression/) 是两个相邻但不同的问题。前一篇讨论的是高频 telemetry 在离开 collector 之前，什么信息可以压缩、什么证据必须留下；这一篇从下一步开始：**原先承诺要保留的证据，因为 loss、probe coverage 或 schema drift 没有被可靠观察到时，collector 应该怎么办？**

## 为什么全局 loss rate 不能直接代表诊断可信度

假设一个 Profiler 需要区分三种慢请求：

1. task 已经 runnable，但长时间没得到 CPU；
2. task 睡在 futex 或其他同步原语上；
3. task 正在等待网络或存储完成。

它可以采 CPU stack，跟踪 scheduler transition，观察一部分 futex 路径，再把 socket 或 I/O completion 与 request identity 关联起来。现在假设 scheduler 事件只丢了 1%。

如果这 1% 接近随机分布，runnable delay 可能仍然估计得很准。但如果 loss 集中发生在事件最密集的一小段 burst，恰好删掉了一次 `runnable -> blocked` transition，同样的 99% coverage 会让 run-queue attribution 完全改变。

Probe 缺失会更麻烦。一个 BPF program 可以一直 attached，也一直产生 record，但新 kernel 或应用版本把某条重要路径移到了它看不到的位置。此时 surviving hook 的 transport coverage 甚至可以是 100%，语义覆盖仍然是不完整的。

固定长度 payload 则形成第三类 blind spot。Collector 收到了每次事件，但只检查前 N bytes。如果 request identity、trace context 或 status 恰好位于 N 之后，它得到的不是随机抽样，而是与 workload shape 强相关的系统性缺失。

因此，更有用的单位不是“整体保留了多少 telemetry”，而是：**某个结论为了排除其他合理解释，需要哪些证据，这些证据现在是否仍然成立。**

## 自适应采集已经存在，但控制目标通常不是“这条诊断还成立吗”

Adaptive observability 并不是新概念。[ViperProbe](https://ieeexplore.ieee.org/document/9335808/) 已经展示过基于 eBPF 的 dynamic sampling 和 workload-aware metric selection。2026 年的 [SysOM-AI](https://arxiv.org/abs/2603.29235) 则把 eBPF tracing、CPU stack、GPU kernel 与 NCCL 事件组合成持续运行的跨层诊断系统；其论文摘要报告低于 0.4% 的 overhead，并称系统已经在超过 80,000 张 GPU 的环境运行，帮助定位 94 个确认的生产问题。

这些工作说明：采集策略可以动态变化，也可以在低 overhead 下支持实际诊断。但它们同时留下了一个更窄的 runtime 问题。很多 adaptive policy 的触发条件是 workload phase、异常、资源预算或固定 diagnosis workflow，而不是“当前这一个结论依赖的证据已经坏了”。

比较实用的第一步，不应该直接发明一个复杂的概率模型，而是把每个诊断放进三个可检查状态：

- **supported**：当前需要的证据都满足声明的 coverage 或 error bound；
- **degraded**：部分证据缺失，但存在一个有界 fallback，仍能区分主要候选解释；
- **unavailable**：现有证据已经无法区分主要候选解释。

`degraded` 不能只是 dashboard 上的一个黄灯。Operator 应该能继续看到：哪一个 evidence obligation 失败了，当前 runtime 正在尝试什么恢复动作。

## 一条 eBPF 诊断到底要跟踪什么

以“慢请求主要来自 run-queue delay”为例，可以把证据依赖写成类似下面的关系：

```text
obligation scheduler_residency:
    hooks = sched_switch, sched_wakeup
    identity = task_generation
    required_coverage = complete transition pairs
    transport = ringbuf_generation_17

obligation blocking_alternative:
    hooks = futex_wait, selected_io_completion
    identity = task_generation
    required_coverage = at least one distinguishing edge

conclusion runqueue_delay:
    requires = scheduler_residency + blocking_alternative
```

真正实现时不一定要使用这种 DSL，但 runtime 至少需要保存等价的信息。对每一个 obligation，可以记录：

- 哪个 BPF program 与 attach generation 负责产生证据；
- 预期 hook 是否仍 attached，record schema 是否兼容；
- 在能够计算时，eligible events 与成功观察 events 的关系；
- ring-buffer reservation failure、perf-buffer loss、map insert failure 与相关 eviction；
- resource semantics 或 schema 的 generation；
- collector restart epoch，以及 epoch 之间是否存在空洞；
- bounded capture 或 parser 条件是否让某条 record 语义不完整；
- 哪些候选 root cause 依赖当前缺失的 signal。

最后一条是普通 telemetry health 最容易缺失的部分。丢一个 futex event 与丢一个 DNS event，不应该自动让所有诊断一起降级。影响取决于当前到底在回答什么问题。

## 为什么一个 `confidence=0.83` 往往不是好抽象

把各种健康信号揉成一个 0 到 1 的 confidence score 很诱人，dashboard 也很容易展示。但这个数字通常很难解释。

10% 随机 sampling loss、一个完全 missing 的 hook、过期的 application schema，以及固定 parser truncation，不存在天然正确的加权方式。即使算出 0.83，也无法告诉 operator 应该补哪一个 probe，或者哪一种结论现在必须停止使用。

更稳妥的第一版应该保留 failure mode 与 obligation 本身，只对确实存在统计模型的采样过程使用概率。很多现实 loss 甚至不是随机的：事件 burst 最容易让 buffer 满；cardinality 暴涨时 bounded map 最容易 eviction；短命进程可能在 discovery attach 之前就结束；大 payload 才会触发 parser limit。这些都是结构化 missing data。

## 现有研究还缺什么

### 1. Loss accounting 往往属于 transport，而不是具体诊断

perf buffer 能告诉 userspace 有 record lost，ring-buffer producer 也可以自己统计 failed reservation，这些都很有用。

但 scheduler event 丢失以后，CPU utilization、run-queue residence、off-CPU attribution 与 request causality 哪一些还能用，并不是 transport 自己知道的。现在常见的做法是抛出“1.2% events dropped”这样的全局 health signal，再让每个下游分析自己判断后果。

缺少的是从 evidence obligation 到 diagnosis 的机器可读依赖。一个有区分度的实验应该逐类注入 event loss，检查系统是不是只禁用真正因此变得 ambiguous 的结论，而不是整个 Profiler 一起失效。

### 2. Probe 还在运行，不代表语义覆盖仍然完整

Kernel path、应用版本、protocol layout 或 compiler transformation 都可能变化。Probe attachment 还正常、record 还在流动，只能证明 collector 活着，不能证明原先的 semantic assumption 仍然成立。

OBI 最近的 bounded capture 和 transport limit 问题说明，有效业务活动可以越过观测边界，而整个 agent 并不会因此停止工作。

这里缺少一个能够独立失败的 coverage contract。Benchmark 应该在保持用户可见 workload 相同的情况下替换 kernel、应用 build、message size 与实际代码路径，检查 Profiler 能不能发现某条诊断的语义前提已经失效。

### 3. Adaptive collection 通常没有明确的“恢复哪份证据”目标

发现异常以后打开更多 tracing 并不等于解决证据缺口。如果当前 ambiguity 是 scheduler 还是 futex，增加 filesystem probe 只会浪费预算；如果问题是 stale resource schema，提高 sampling rate 也无法恢复 identity；如果决定性事件早已发生，事后开启详细 trace 甚至来不及。

缺少的是针对 failed obligation 的 recovery plan。它应该和 generic verbose mode 在同样 CPU、memory、export budget 下直接比较，看是否能用更低开销恢复同样的判断能力。

### 4. Profiler 需要把“拒绝回答”作为一等结果

Observability 产品天然倾向于给出答案，因为 `unknown` 看起来不如一个明确 root cause 有价值。但自动 remediation 与 AI-assisted operations 会把诊断直接变成系统动作，例如迁移 workload、调 limit、重启服务或 rollback。

如果当前证据无法区分两个主要解释，`unknown because sched_switch coverage was incomplete during generation 17` 比一个看起来很专业但实际上不受支持的结论更有工程价值。

## 兼具学术价值与生产价值的方向

### 1. 把 diagnosis obligation 编译成 evidence-deficit ledger

**缺口。** 现有 collector 可以报告底层 loss 与 health，却很少知道具体诊断依赖哪些 signal。

**机制。** 在前一篇 diagnostic contract 的基础上，把每个 conclusion 编译成一张小型 dependency graph。每个 obligation 带上 hook generation、identity generation、loss accounting、schema version 与允许的 approximation mode。BPF program 只维护便宜的局部事实，例如成功 observation、failed reservation、map pressure、generation ID 与 invariant violation；userspace 再把这些事实与 attach state、semantic metadata 组合成 `supported / degraded / unavailable` ledger。

**与已有工作的差别。** loss callback 描述 transport，diagnostic compression contract 描述一个 compact representation 承诺保留什么。这里把两者连起来，让具体 query 知道自己的 prerequisites 是否仍然成立。

**Artifact。** 一个小型 contract language、一组 libbpf-side accounting helper、userspace evidence ledger，以及几个现有 eBPF Profiler 的 adapter。

**评测。** 准备 scheduler、network、memory 与 application-resource 的 ground-truth incident，分别注入 ring-buffer pressure、perf loss、map eviction、missing hook、collector restart 与 schema mismatch。比较无 health metadata、只有 global loss counter、以及 obligation ledger 三类系统，测 root-cause accuracy、false confident diagnosis、正确 abstention、diagnosis availability 和 overhead。

**学术价值。** 核心问题是 observability completeness 能否相对于“一个诊断”定义，而不是只相对于“一个 transport stream”定义。

**生产价值。** Fleet operator 可以在部分 telemetry failure 期间继续安全使用仍然有证据的结论，而不必在“全关 Profiler”和“全信剩余数据”之间二选一。

**失败条件。** 如果简单的 global loss + attach-health threshold 在不同 incident 中已经能同样准确预测错误诊断，这张 dependency graph 就没有必要。

### 2. 按 evidence obligation 恢复，而不是把整个系统切到 verbose mode

**缺口。** 很多 collector 降级以后只有两个选择：照常运行，或者突然打开大量详细 tracing。

**机制。** 某个 obligation 进入 `degraded` 后，userspace controller 只选择与该缺口相关的 bounded recovery action，例如降低一个 event family 的 sampling divisor、临时启用 fallback tracepoint、为一个 entity generation 增加 raw exemplar、刷新 resource-semantics manifest，或者给受影响 stream 多分配一小段 export budget。

Controller 需要 hysteresis 和硬预算，也需要知道什么时候已经无法恢复。如果决定性 transition 已经过去，而且没有 pre-trigger exemplar，正确状态应该直接进入 `unavailable`。

**与已有工作的差别。** ViperProbe 等系统已经证明动态 collection 可行。这里的 trigger 与 target 更具体：因为某个声明过的 evidence obligation 失败才改变采集，恢复后停止；如果恢复不了，就明确 abstain。

**Artifact。** 一个 userspace policy controller，加上一组 fallback probe plan。对于需要 late dynamic attachment 的场景，同一 contract 也可以驱动 [bpftime](https://eunomia.dev/zh/bpftime/) 之类的 userspace eBPF runtime。

**评测。** 在不同 event rate 与 blind-spot 类型下回放 incident，对比 static low-overhead、always-high-fidelity、anomaly-triggered verbose mode 与 obligation-targeted recovery，并固定 CPU、map memory 与 export budget。主要指标包括恢复 supported diagnosis 的时间、无效 probe 工作量、missed root cause 与 workload perturbation。

**学术价值。** 这把 adaptive observability 从一个通用 fidelity knob 变成“在资源约束下修复缺失证据”的控制问题。

**生产价值。** Fleet Profiler 只在证据不足的位置增加开销，避免一次异常让整个集群突然产生 trace storm。

**失败条件。** 如果 targeted recovery 经常和 high-fidelity mode 一样贵，或者总是在决定性事件之后才触发，简单静态模式更合理。

### 3. 专门测 confident-but-wrong 的 observability benchmark

**缺口。** 很多 Profiler benchmark 会测 overhead，也会在 telemetry 正常时检查能不能诊断已知 incident，却很少把“证据坏了以后还自信地答错”作为主要 failure。

**机制。** 构造具有相同外部症状、但真实 root cause 不同的 paired workload，然后主动破坏 observer，而不只破坏被观测应用。每个 case 保存完整 ground truth，并覆盖：

- ring-buffer reservation failure 与 perf-buffer loss；
- 短命进程造成 missed attach；
- kernel 或应用版本变化导致 hook coverage 改变；
- bounded-map eviction 与 identity reuse；
- payload 超过 parser 或 transport capture limit；
- collector restart gap；
- stale semantic manifest；
- root-cause transition 周围的非随机 burst loss。

Benchmark 不只问“root cause 对不对”，还问 Profiler 能不能知道自己已经没有足够证据选择。

**与已有工作的差别。** 常见 observability 评测重点是 runtime cost、trace volume 或 diagnosis success。这里把 **observer degradation 下的 false confidence** 作为第一指标。

**Artifact。** 可复现 Linux workload、fault injector、ground-truth trace，以及同时评估 diagnosis accuracy、abstention calibration、recovery time 与 resource cost 的 harness。

**评测。** 比较 full tracing、fixed sampling、手写 eBPF aggregation、只看 loss counter 的 diagnosis，以及 obligation-aware adaptive collection。通过移除 query-to-evidence dependency 的 ablation 检查新增语义到底有没有价值。

**学术价值。** 它把“能不能相信 Profiler”从抽象讨论变成可测系统性质：在结构化证据缺失下，系统能否区分“我判断错了”和“这个问题现在根本回答不了”。

**生产价值。** Observability 团队可以在新 kernel、collector 升级或预算调整上线前，专门 regression-test 这种 confident-but-wrong failure。

**失败条件。** 如果真实 incident 很少因为这些 evidence fault 改变诊断结果，显式 validity control 的收益就不足以覆盖复杂度。

## 最终运行规则可以很简单

eBPF Profiler 不应该只问“我收到了多少 telemetry”，而应该问：“为了排除我声称已经排除的其他解释，我需要的证据现在还在吗？”

这个规则会改变几项实现细节。Loss metadata 需要进入 query semantics；attach generation 与 schema generation 需要进入 diagnostic state；adaptive collection 有明确的恢复目标；`unknown` 也成为合法输出。

这不要求第一版系统对每一个 root cause 做形式化证明。一个有用的最小版本可以只覆盖几个高价值诊断：声明需要哪些 evidence，跟踪已知 deficit，在 contract 被破坏时停止过度解释。

## 哪些结果会改变这个判断？

本文假设结构化 evidence loss 足够常见，确实会改变线上 diagnosis，而且普通 collector health 太粗，无法预测哪些具体结论因此失效。

如果大规模生产数据表明，只靠 global loss threshold、attach-health check 与静态 safety margin，就已经能识别几乎所有不可靠 diagnosis，那么 obligation graph 的维护成本不值得承担。

另一个可能推翻结论的结果，是 targeted recovery 总是反应太慢。很多 incident 的决定性 transition 发生在可见 symptom 之前。如果 bounded exemplar 与 fallback probe 也无法保住这些 transition，adaptive control 就应该更多选择 abstain，而不是承诺“之后再补采”。

最有说服力的正向证据，是让两个 Profiler 接收同样的 degraded telemetry，并保持相近 overhead 与数据量：obligation-aware 版本显著减少 confident wrong diagnosis，同时在必要证据仍完整的 case 上保持可用。如果这个结果无法跨 scheduler、network、memory 与 application-semantic incident 重复，diagnosis validity 可能仍然只是一个更简单的 collector-health 问题。

## 参考资料

- Linux kernel documentation, [BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html)。
- libbpf documentation, [`perf_buffer__new()` 与 lost callback](https://libbpf.readthedocs.io/en/latest/api.html)。
- OpenTelemetry eBPF Instrumentation, [超过 16 KiB 的合法 Go Auto SDK span 会越过当前 BPF transport 上限](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2958)，2026-08-07。
- OpenTelemetry eBPF Instrumentation, [缺失 response observation 时可能把真实 200 请求合成为 HTTP 499](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/3067)，2026-08-17。
- OpenTelemetry eBPF Instrumentation, [`traceparent` 超出 capture buffer 时可能造成 trace context 断裂](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/1381)，2026-02-28。
- M. H. Abbasi et al., [ViperProbe: Rethinking Microservice Observability with eBPF](https://ieeexplore.ieee.org/document/9335808/), IEEE CloudNet 2020。
- Yusheng Zheng et al., [SysOM-AI: Continuous Cross-Layer Performance Diagnosis for Production AI Training](https://arxiv.org/abs/2603.29235), 2026。
