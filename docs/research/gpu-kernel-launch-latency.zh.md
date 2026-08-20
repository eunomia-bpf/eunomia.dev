---
date: 2026-08-20
title: "GPU 内核是真的慢，还是只是启动晚了？"
description: "GPU 启动延迟可能来自主机调度、运行时、命令队列、依赖关系或设备资源等待。本文拆开这些阶段，并提出可验证的因果时间线和基准。"
tags:
  - Daily Report
  - GPU Profiling
  - CUDA
  - Observability
research_question: "性能分析器怎样区分主机侧延迟、命令缓冲区排队、依赖等待和 GPU 内核本身的执行时间，而不从一段时间线空隙里臆测因果关系？"
source_cutoff: 2026-08-20
status: daily-report
---

# GPU 内核是真的慢，还是只是启动晚了？

一条 CUDA 内核的时间线看起来很简单：CPU 线程调用 launch API，过一段时间，GPU 上的内核开始执行，随后结束。但如果 API 在 10.000 ms 开始，而内核到 12.000 ms 才启动，中间缺失的两毫秒并不是同一种延迟。

CPU 线程可能在调用 CUDA 之前就被调度出去了；运行时或驱动可能在 API 内部花了时间；launch 命令可能已经写进 command buffer，却还没有提交给 GPU；它也可能因为 stream、event 或 CUDA Graph 的依赖而必须等待。即使依赖已经满足，GPU 上还可能有其他工作或资源压力，导致新 kernel 无法马上执行。

这些情况的修复方向完全不同，但性能分析器很容易把它们压成一个叫做 **launch latency** 或 **queue time** 的数字。

NVIDIA 现有工具其实已经提供了相当丰富的原始证据。CUPTI 的 API activity 记录包含 API 的开始和结束时间、进程和线程身份，以及能和对应 kernel activity 对上的 correlation ID。当前的 kernel activity 还可以提供 `queued` 和 `submitted` 时间戳，只是这类 latency timestamp 默认并不采集。Nsight Systems 把 CUDA kernel 的一次 launch 分成 API time、queue time 和 kernel time，同时明确说明一个容易被忽略的限制：它报告的 queue time 从 CUDA API 返回时刻算到 kernel 开始时刻，但真正的 enqueue 发生在 API 调用内部，因此这个区间只是近似值。kernel 甚至可能在 API 返回之前就已经启动，此时这个简化的 queue time 不会出现。

所以现在缺的不是“再加一个时间戳”。更窄、也更有价值的问题是：**性能分析器能不能把这些时间戳和身份信息组合成一个站得住脚的解释，说明某次 kernel 为什么启动得晚？**

这不是一个必须依赖 eBPF 才能成立的问题。在 Linux 上，eBPF、perf、ftrace 都可以补充主机侧调度证据，但本文讨论的核心是 GPU profiling 和因果归因；换成其他操作系统或 host tracer，这个问题仍然存在。因此本文按相邻系统方向分类，而不是 eBPF-centered 报告。

## 现有 GPU trace 已经能告诉我们什么

CUPTI 已经提供了一条很有用的 host-to-device 关联链。一次 runtime 或 driver API 调用有 start/end timestamp 和 `correlationId`，对应的 kernel 记录会携带相同的关联身份。对于 CUDA Graph，kernel activity 还可以带 graph 和 graph node 的身份。CUPTI 的 external correlation 还能把上层 runtime 或应用自己的 ID 映射到 CUDA correlation ID，不过这种 external-correlation stack 是按 CPU thread 维护的，映射关系也需要 client 自己管理。

对一次 kernel launch 来说，当前 CUPTI kernel record 最有价值的四个时间点是：

1. `queued`：launch 命令被写入 CUDA command buffer；
2. `submitted`：包含该 launch 的 command buffer 被提交给 GPU；
3. `start`：kernel 真正开始执行；
4. `end`：kernel 执行结束。

有了这些信息，问题已经可以比“kernel 跑了多久”细很多。比如，API 进入到 `queued` 很长，和 `queued -> submitted` 很长显然不是一类现象；`submitted -> start` 很长又是另一回事。

但这些依旧只是**事件边界**，不是完整的状态机。`submitted` 并不等于“所有依赖都已经满足，现在只是在等 GPU scheduler”。CUDA stream 顺序、event、graph dependency、同步、资源占用都会影响一次 kernel 什么时候才允许开始执行。新的 Programmatic Dependent Launch 还进一步说明了这个问题：CUDA 可以允许 dependent kernel 提前和前序 kernel 重叠，但官方文档明确说明，这种并发只是机会而不是保证，依赖这种并发行为本身是不安全的。

Nsight Systems 因此采取了一个合理的工程折中：把 launch 粗分成 API、queue 和 kernel 三段，并明确提醒 queue time 本身并不代表坏事。GPU 正在高效执行其他工作时，新 kernel 有 queue time 反而很正常。

真正缺少的是一层解释机制：它要保留这些差异，而不是把所有 kernel 启动前的空隙都归到同一个原因上。

## 诊断上的根本问题：时间区间不是原因

假设一个模型服务会反复 launch 同一个约 80 微秒的 CUDA kernel。某次部署之后，请求延迟增加了 1.5 ms。trace 里 kernel 自己仍然只执行约 80 微秒，但是 start time 整体往后移了。

至少有四种完全不同的解释：

- **主机就绪延迟。** 应用或框架没有及时走到 CUDA API，因为 launch 所在 CPU thread 被抢占、阻塞，或者正在处理其他任务。
- **运行时或 command-buffer 延迟。** CUDA API 本身变慢，或者从 API 进入到 command queue / submit 的路径变慢。
- **依赖等待。** kernel 按语义就应该排在 stream、event、graph 或 programmatic dependency 后面。这种时间不能叫 GPU scheduler delay。
- **设备可用性等待。** launch 已经提交，而且可以执行，但更早的 GPU 工作或资源占用让它没法马上开始。

还有第五种情况更重要：trace 根本没有足够证据区分“依赖还没满足”和“GPU 已经可以执行但暂时没有资源”。可信的性能分析器应该明确说 **这个边界上无法判断**，而不是根据时间线里颜色最像的一段猜一个原因。

这个区别会直接改变工程决策。绑核不能修复一个合理的 stream dependency；重写 CUDA Graph 也不能修复一个晚醒 2 ms 的 CPU thread。kernel duration 已经稳定时，继续优化 kernel 本体没有意义。更高的 GPU 利用率甚至可能增加 queue time，同时改善整体 throughput。

所以 profiler 真正需要回答的是：**哪些状态转换有直接证据，哪些只是有条件推断，还有哪些原因在当前观测下无法区分？**

## 现有研究还缺什么

### 1. 缺少统一的 launch 状态语义

CUPTI 已经提供了构建更细时间线的大部分原料，但不同工具仍然需要自己解释每一段 interval 的意义。`queued -> submitted` 有比较明确的 command-buffer 语义；而 `submitted -> start` 更复杂，因为 dependency、已有 GPU workload、资源可用性等因素都可能参与其中。

真正缺少的不是新 timestamp，而是一份机器可读的状态契约：某个 timestamp 证明了什么状态，哪些前置条件已经知道，哪些仍然未知。最直接的验证方法是做一个可以独立注入各种 delay source 的 benchmark。如果 profiler 找对了 interval，却经常把 cause 标错，说明状态契约还不够。

### 2. CUDA correlation 不等于上层工作在 host 侧的完整因果链

CUPTI correlation ID 很擅长把 CUDA API 和 GPU activity 连起来。External correlation 还能继续接到其他 API domain，但映射由 client 管理，而且上下文本身是 CPU-thread scoped。现代 runtime 可能在一个线程准备工作，交给另一个线程池，随后 batch 成 graph，最后又由另一个基础设施线程完成 launch。

这里缺少的是能够穿过 host handoff 和 graph transformation 的稳定 **launch identity**，而不仅是某个 API 周围的一次 join。Profiler 应该能区分：“真正的业务工作早就 ready 了，但是提交线程醒晚了”，和“应用本身就晚到这个提交线程”。验证可以通过线程池搬运、graph replay 和已知 request ID 的 workload 来完成。

### 3. Queue summary 不能证明 dependency 已经 ready

Nsight Systems 很清楚地说明，它的 queue time 是从 API end 到 kernel start 的近似区间，而且 queueing 本身未必是问题。CUPTI 的 `queued` / `submitted` timestamp 能把 command-buffer 边界切得更细，但这些字段仍然不能单独证明“所有 dependency 在某一刻已经全部满足”。

缺少的是明确的 readiness boundary；如果这个边界本身不可见，就至少应该有 uncertainty marker。验证时可以分别构造已知的 stream/event/graph dependency，以及独立的 GPU saturation。如果一个 profiler 会把合法的 dependency wait 归因成 scheduler delay，这个诊断就不可信。

## 兼具学术价值与生产价值的方向

### 方向一：带显式未知状态的 launch-state ledger

**缺口。** 现有工具提供了许多时间戳，却缺少从应用意图到 GPU 执行之间统一、可验证的状态解释。

**机制。** 为每个逻辑 launch 建立一个 append-only ledger，只记录当前最强证据：

```text
request_ready? -> api_enter -> queued? -> submitted? -> dependency_ready? -> kernel_start -> kernel_end
```

这里的问号不是实现不完整，而是 schema 的一部分。每个字段需要标明它是直接观测、在某个具名规则下推断，还是未知；每个 transition 还记录证据来源，例如 runtime hook、CUPTI API activity、CUPTI latency timestamp、graph dependency、OS scheduler trace 或 application correlation。

如果 runtime 无法给出真正的 `dependency_ready`，ledger 不应造一个时间点，而应保留区间约束，例如 `submitted <= ready <= start`。这样诊断结果可以说“提交到执行之间有 1.2 ms，但 dependency readiness 不可见”，而不是武断地说“GPU scheduler delay 是 1.2 ms”。

**和现有工作的区别。** 这不是替代 CUPTI 或 Nsight Systems，而是在已有 activity record 上增加一层保留不确定性的语义解释。

**可实现产物。** 一个开放的 trace schema，加一个 CUPTI reference collector；只在需要时开启 latency timestamp。在 Linux 上，host scheduling 可以来自 perf、ftrace、eBPF 或 Nsight 的 OS runtime/context-switch trace，schema 本身不绑定某一种 collector。

**评估。** 在可控注入不同 launch delay 的 workload 上测量阶段边界误差和 cause classification precision，对比普通 API/queue/kernel summary 以及 full-trace baseline。同时报告采集开销，以及有多少 launch 最终被诚实标记为 unresolved。

**学术价值。** 一般化问题是：多个 scheduler 和 dependency system 共享一条异步时间线时，怎样表达可证明的因果关系和不可消除的不确定性。

**生产价值。** 用户不再需要手工同时看 CPU scheduler、CUDA API 和 GPU timeline，工具可以直接指出问题更可能位于 host、runtime、dependency graph 还是 device boundary。

**失败条件。** 如果现有 CUPTI latency timestamp 加 Nsight report 已经能以同样准确率、较低开销识别所有注入原因，那么额外 ledger 没有必要。

### 方向二：能穿过线程切换和 Graph batching 的跨域 launch identity

**缺口。** CUDA correlation 在 CUDA API 附近很精确，但上层工作可能跨 CPU thread，或者在最终 launch 前被转换成 CUDA Graph node。

**机制。** 在应用或 framework 第一次声明某段 GPU work 时创建一个 versioned `launch_epoch`。它随 host handoff 传播；等到 CUDA boundary 出现后，再绑定 CUPTI external correlation、CUDA correlation ID，以及 graph/node ID。映射必须允许一对多，而不是假设一个 request 只对应一个 kernel。

关键属性不是 ID 的名字，而是**lineage**。同一个 graph node 被 replay 多次时，需要同时保留稳定 node identity 和每次 replay 的 launch epoch；一个 request fan-out 到多个 stream 时，也要保留这种分叉，而不是压平到一个 thread-local stack。

**和现有工作的区别。** External correlation 已经提供了连接点；这里新增的是 thread-pool handoff、graph construction/replay 和 batched launch 之间的传播语义。

**可实现产物。** 一个小型 correlation library、一个 framework adapter 和一个 trace validator。第一版可以针对“host thread pool + CUDA Graph replay”的应用，完全不需要修改 GPU driver 就能生成 ground truth。

**评估。** 注入 thread handoff、graph replay 和 batching，测 lost join、wrong join、存储开销和最终诊断准确率。对比纯 thread-local correlation、仅 CUDA correlation，以及 launch-epoch 方案。

**学术价值。** 可以验证一份很小的 lineage contract 是否足够让 host/device heterogeneous trace 在异步 runtime 边界上具有可组合性。

**生产价值。** Framework 团队可以把一个用户请求准确关联到真正启动晚的 GPU work，即使最后的 CUDA call 是由基础设施线程发出的。

**失败条件。** 如果真实 framework 中现有 CUDA/graph correlation 已经天然保留了所需 lineage，那么 launch epoch 只是重复信息。

### 方向三：以真实原因作为 ground truth 的 launch-delay benchmark

**缺口。** Profiler 可以画出非常精细的 timeline，却通常没有证明“它对 pre-kernel gap 的解释是对的”。

**机制。** 构造一套一次只注入一种 delay source 的 workload，并在 profiler 之外记录真实原因：

- CUDA API 之前的 host descheduling；
- launch path 中人为增加的 CPU work；
- command-buffer batching 或 submission delay；
- 显式 stream/event dependency；
- CUDA Graph dependency 和 replay；
- 独立 stream 或进程制造的 GPU saturation；
- 不同 resource footprint 的 kernel；
- Programmatic Dependent Launch 中“允许重叠但不保证重叠”的场景。

之后再把多个原因组合起来，测试真正的歧义情况。Benchmark 还应同时包含 latency-sensitive 的短 kernel 和以 throughput 为目标、正常 queueing 很多的 workload。

**和现有工作的区别。** 很多 profiler benchmark 会测 timestamp 是否准确、采集是否便宜；这里测的是：**工具从这些 timestamp 推出的解释是否准确。**

**可实现产物。** 一组可重复执行的 CUDA workload、独立 ground-truth label、CUPTI/Nsight export adapter，以及对 interval error、cause accuracy、unresolved calibration 和 overhead 的 scorer。

**评估。** 最强 baseline 就是开启当前 CUPTI/Nsight 能提供的最丰富 trace。新的设计必须减少错误因果诊断，而不能仅仅增加事件数量。Ablation 可以分别去掉 host scheduling、queued/submitted timestamp、dependency metadata 和 launch lineage，直接量化每类证据的价值。

**学术价值。** 它把 heterogeneous scheduling attribution 从“看时间线的经验判断”变成可证伪的测量问题。

**生产价值。** Profiler 和 framework 团队可以在发布新的“launch latency”诊断前，先确认它真的会把用户引导到正确子系统。

**失败条件。** 如果简单的 API/queue/kernel 三段划分已经能稳定分类所有注入场景，更复杂的因果模型就不值得维护。

## 这会怎样改变 profiler 的输出

有用的输出不应该只是更细的彩色时间线，而应该是一组语义更强的判断：

- 应用本身晚 ready；
- CUDA API path 花费异常；
- launch 在提交前有可观测等待；
- execution 被某个已知 dependency 合法阻塞；
- launch 已经 submitted，但 readiness 无法确定；
- kernel 自己执行确实变慢。

Profiler 应该优先给出这些判断，或者明确的 unresolved interval，而不是笼统提示“GPU launch latency 很高”。

这和最近几篇 Daily Report 是互补关系，而不是重复。[采样偏差报告](https://eunomia.dev/zh/research/profiler-sampling-bias/)问的是测量结果在统计上是否可信；[异步 eBPF 因果 profiler 报告](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/)问的是逻辑工作怎样跨 CPU 侧异步 handoff。本文问的是另一个 GPU boundary 问题：当 host/device event 都已经看得见时，还需要什么证据才能解释**为什么执行在这个时刻才开始**？[页面级内存归因报告](https://eunomia.dev/zh/research/page-level-ebpf-memory-attribution/)也体现了同一个更宽泛的原则：观测到 activity，并不等于已经知道它的 causal ownership。

## 哪些结果会改变这个判断？

有三类结果会明显缩小甚至推翻本文对新归因层的需求。

第一，如果当前 CUPTI latency timestamp、graph identity 和 Nsight Systems 的 host scheduling trace 已经能在上述注入 benchmark 中以很高 precision 区分原因，而且 overhead 可接受，那么缺的可能只是文档或 UI，而不是新的 trace contract。

第二，如果现代 CUDA workload 的 dependency readiness 根本无法被可靠观测或有效约束，那么 profiler 就不应该继续细分所谓 device delay。更正确的产品可能只是展示 command-buffer timing，再明确标记一段无法区分 dependency/device 的区间。

第三，如果开启 `queued` / `submitted` timestamp 或跨域 lineage 会明显扰动 latency-sensitive workload，甚至改变它原来的 scheduling behavior，那么设计应该分层：常态只保留廉价的 API/kernel correlation，在少量选中 launch 或短诊断 epoch 中再打开丰富证据。

目前的证据支持一个相对克制的结论：GPU profiler 已经拥有分析 launch delay 所需的很多时间戳，但**拥有时间戳，还不等于拥有可信的因果诊断**。下一步最值得做的，是把这两者之间的差距做成可以被 benchmark 直接验证的问题。

## 参考资料

- NVIDIA，CUPTI Activity API，API activity 与 kernel activity 字段：<https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html>
- NVIDIA，`CUpti_ActivityAPI`：<https://docs.nvidia.com/cupti/api/structCUpti__ActivityAPI.html>
- NVIDIA，CUPTI usage guide，external correlation：<https://docs.nvidia.com/cupti/main/main.html>
- NVIDIA，CUPTI `CUpti_ActivityKernel9`，包括可选的 `queued` / `submitted` latency timestamp：<https://docs.nvidia.com/cupti/13.0.0/api/structCUpti__ActivityKernel9.html>
- NVIDIA，Nsight Systems Post-Collection Analysis Guide，CUDA kernel launch/queue report：<https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html>
- NVIDIA，CUDA Programming Guide，Programmatic Dependent Launch and Synchronization：<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>
