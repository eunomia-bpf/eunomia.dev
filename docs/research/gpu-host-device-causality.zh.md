---
date: 2026-08-20
title: "GPU Kernel 变慢时，Profiler 能证明是谁造成的吗？"
description: "GPU 时间线可以看到 kernel 变慢，却未必能证明是哪次 host 操作造成延迟。本文分析异步 CUDA 的关联缺口，并提出可验证的 host-device 因果契约。"
tags:
  - Daily Report
  - GPU
  - Profiling
  - CUDA
  - CUPTI
  - Causality
research_question: "生产环境中的 GPU profiler 怎样把 host 侧工作经过异步 CUDA stream 与 graph 一直关联到 device activity，并避免仅凭时间相邻关系推断因果？"
source_cutoff: 2026-08-20
status: daily-report
---

# GPU Kernel 变慢时，Profiler 能证明是谁造成的吗？

一个服务收到请求 R42。CPU 线程先准备 batch，再把任务提交到两个 CUDA stream，然后马上继续执行 host 侧代码。50 ms 之后，其中一个 GPU kernel 比平时晚启动。时间线上恰好还能看到一次 CPU 调度空档、一次显存拷贝、几次 kernel launch 和这个晚启动的 kernel。

到底是哪一个 host 侧动作造成了延迟？

时间线可以给出线索，但在异步 GPU 执行里，时间接近并不能证明因果。CUDA API 可能在 device 工作真正开始前就返回，不同 stream 中的命令可以并发，跨 stream 的 event 会建立显式依赖，CUDA Graph 会反复执行预先构建好的依赖图，而 device scheduler 也可能因为资源占用延迟一个本来独立的 kernel。一个 host event 出现在 kernel 前面，不代表它就是这个 kernel 的原因。

<!-- more -->

现有接口其实已经提供了不少关联信息。NVIDIA CUPTI 会为 CUDA driver 或 runtime API 调用分配 correlation ID，并把相同 ID 放进对应的 kernel、memcpy 和 memset activity record。它还支持 external correlation ID、CUDA Graph 的 `graphId` 与 `graphNodeId`、stream/context 标识、dropped-record 计数，以及可选的 kernel queued/submitted timestamp。当前 CUDA Programming Guide 则把 stream 定义为有序 work queue，并允许 event 显式建立跨 stream 依赖。

这些机制可以把局部事件连起来，却还不足以回答一个跨越应用任务、host 调度、CUDA API、stream dependency 与 GPU 执行的生产根因问题。真正缺少的是一种能够沿整个路径保持稳定，并且在证据不完整时明确承认缺口的因果身份。

本文主张引入一个 **host-device 因果契约**：把 GPU profiling 表示成类型化依赖图，把稳定的逻辑工作身份带过 host/device 边界，用 CUDA stream、event 与 graph 语义建立硬依赖，并在证据缺失时保留 `unknown`，而不是用 timestamp 猜出一条看似完整的链。

这篇报告归类为 adjacent systems，而不是 eBPF-centered。eBPF 很适合观察 host 调度、进程、page fault、driver interaction 和闭源应用边界，但同一个因果机制也可以使用其他 host tracer；CUPTI 与 CUDA 自己的依赖语义同样是核心组成部分。

## CUDA 已经能做局部关联，但局部关联还不是端到端因果

CUPTI correlation ID 解决了一个重要问题。CUDA API 调用发起 kernel 或 memory operation 后，API activity 与对应 GPU activity 可以携带同一个 correlation ID。Profiler 因而能回答“这个 kernel record 是哪次 `cudaLaunchKernel` 产生的？”

External correlation 又能向上一层扩展。CUPTI 可以把外部 ID 关联到 CUDA API activity，用来连接更高层 region 或 API 与其中发出的 CUDA 工作。CUDA Graph activity 还带有 graph 与 graph-node identity，stream ID 则保留了工作被提交到哪一个执行队列。

仓库里已经有 [CUPTI correlation 示例](https://eunomia.dev/others/cupti-tutorial/cupti_correlation/) 和 [external correlation 示例](https://eunomia.dev/others/cupti-tutorial/cupti_external_correlation/)，说明底层 join 本身可以实现。真正困难的问题从它们上一层开始。

假设请求 R42 先由线程 T1 处理，再把预处理交给 T2。T2 被调度器暂停了 4 ms，随后调用一个 framework API，而 framework 又通过 executor 在 T3 上调用 CUDA。这个 CUDA API 启动一个包含多个 stream 的 graph。CUPTI 可以把 API record 与 GPU record 连起来，但 profiler 仍然需要知道这些 record 都属于 R42，并且 T2 上的 4 ms 延迟确实位于晚启动 kernel 的依赖路径上。

PID 和 timestamp 不表达这个关系。工作经过 executor、callback、queue、future、CUDA Graph 或库内部 worker thread 后，TID 也不够用。分布式 tracing 有同样的经验：span 的时间只有在系统先知道哪些 span 属于同一条链之后才有意义。

因此，host/device 边界需要一个一等的工作身份，而不是更多时钟。

## 异步执行把总时间线变成了偏序关系

CUDA Programming Guide 对诊断最重要的两个性质是：

1. 同一个 stream 内的操作按照 enqueue 顺序执行；
2. 不同 stream 的操作可以并发，而 CUDA event 和 `cudaStreamWaitEvent()` 可以建立跨 stream 依赖。

所以，多 stream 执行天然是一个偏序。如果 kernel B 等待一个在 kernel A 之后记录的 event，那么 A 完成是 B 的真实前驱。如果 kernel C 与 D 位于独立 stream，之间没有 dependency，那么它们 timestamp 很接近也不能证明其中一个拖慢了另一个。

CUDA Graph 更直接地表达这个事实。Graph 本身就是依赖结构，不只是时间线；CUPTI 还能为 activity 提供 graph 与 graph-node ID。Profiler 如果只把所有记录按时间排序，就丢掉了 runtime 已经知道的因果信息。

这会直接影响根因判断。假设 kernel K 比预期晚 6 ms 启动，至少有三种解释：

- host thread 晚了 6 ms 才提交 K，因为它被 CPU scheduler 挂起；
- K 按时提交，但等待另一个 stream 的前驱；
- K 已经 ready，却因为 device resource 被无关工作占用而没有立刻执行。

三种情况都可能在 K 前面出现相似的空档，但修复方法完全不同。第一种应查 host scheduling 或 CPU contention，第二种应查 dependency construction，第三种才更像 device scheduling 或 interference。

CUPTI 可以选择性记录 kernel 进入 command buffer 的 queued timestamp，以及 command buffer 提交到 GPU 的 submitted timestamp。这些边界非常有价值，但它们仍然只是因果模型里的观测，不是因果模型本身。

## 跨层系统已经证明关联有价值，但还没有统一的因果契约

近期生产系统已经说明跨层 GPU diagnosis 值得做。

[SysOM-AI](https://arxiv.org/abs/2603.29235) 持续组合 CPU stack profiling、GPU kernel tracing、NCCL instrumentation 与 eBPF tracing，并通过分层差分诊断缩小问题范围。论文报告的生产部署覆盖超过 80,000 张 GPU，说明许多原本需要手工拼接多个工具的问题可以在统一证据下更快定位。

[Host-Side Telemetry for Performance Diagnosis in Cloud and HPC GPU Infrastructure](https://arxiv.org/abs/2510.16946) 也把 host-side eBPF telemetry 与 GPU 内部事件结合，用于区分 NIC contention、PCIe pressure 与 CPU interference 等共享基础设施原因。

这些工作说明一个生产事实：GPU symptom 的原因经常不在 GPU 里面。但它们并不会让每一条 host/device 关系自动带上完整因果身份。差分诊断可以发现某个 rank 或 host 与同伴不同，却不一定证明每一个延迟 operation 的准确 parent edge。

本站此前的 [GPU observability 分析](https://eunomia.dev/blog/2025/10/14/gpu-observability-challenges/) 已经讨论过 CPU 与 GPU 工具割裂造成的跨层可见性缺口。本文把问题进一步收窄：一个 profiler 到底需要保留什么，才能让“这条 host 事件导致了这个 GPU 延迟”成为可测试的结论，而不是从图上看起来合理的解释。

## 现有研究还缺什么

### Correlation ID 到应用语义开始的地方就停了

CUPTI correlation ID 是 CUDA API activity 与对应 GPU work 之间很强的局部 join key，却不会自动告诉 profiler：是哪一个用户请求、training step、inference token、runtime task、workqueue item 或 framework operation 发起了这次 API 调用。

Framework 可以加入 NVTX range、external correlation ID 或自己的 task ID，但闭源库与内部 worker thread 会让覆盖不完整。通用 profiler 因此需要在执行路径换线程后仍然保留逻辑工作身份。

验证这个缺口并不复杂：构造一个逻辑请求，让它经过多个 host queue 后才 launch GPU work。如果 profiler 不依赖 timestamp proximity 就无法恢复正确的 request-to-kernel mapping，那么身份契约还不完整。

### Dependency 语义比按 timestamp 排序丰富得多

时间线可以按 start time 排序。CUDA stream、event 与 graph 表达的是 dependency constraint，两者不是一回事。

Profiler 必须区分“发生得更早”和“必须先完成”。否则，一个更早但独立的 kernel 只因为和后一个 kernel 时间接近，就可能被错误归因为 delay source。真正缺少的是一张 dependency graph：有 runtime 语义时使用显式 edge，没有足够证据时保留不确定性。

一个有区分力的实验可以让两组相同 kernel 保持接近的 timestamp，只在其中一组加入跨 stream event dependency。因果 profiler 的解释应该随着 dependency 改变，而纯 timestamp baseline 可能给出同样的结论。

### 丢记录应该打断因果结论，而不是只让图变稀疏

CUPTI 提供 dropped-record 计数，一些 activity timestamp 也可能是 unknown。Activity 通过异步 buffer 交付，host tracer 同样可能丢 sample 或漏掉某个 library boundary。

如果建立一条 causal path 所需的 record 丢失，profiler 应该明确说这条 path 不完整。很多 trace pipeline 仍然会照常绘制剩余事件，让“缺了证据”看起来像“本来就没有 dependency”。

这不是单纯的 telemetry quality 问题，而是 correctness 问题。Operator 必须能区分“dependency 不存在”和“证明 dependency 的 record 丢了”。

### 现有 evaluation 很少直接计算 causal edge 是否正确

GPU profiling 论文通常报告 overhead、diagnosis accuracy 或 time to root cause。这些指标有价值，却可能掩盖一种情况：系统给出了正确 label，但背后的因果链是错的。

如果要把 causal explanation 当产品能力，就需要 parent edge 和 critical-path attribution 的 ground truth。否则系统可能依靠 workload-specific heuristic 提高 diagnosis accuracy，一换 workload 就给出错误解释。

## 兼具学术价值与生产价值的方向

### 1. 以 generation 为边界的 host-device causal token

**缺口。** CUDA correlation ID 连接 API 与 GPU activity，host tracer 连接进程与系统事件，但两边都不保证一个逻辑工作跨越 host queue、thread、CUDA stream 与 graph launch 后仍然有稳定身份。

**机制。** 给每个被 profiling 的 work unit 一个 128-bit causal token，并附带 generation。存在应用边界时，token 从 RPC、training step、inference iteration 或显式 profiling region 开始。工作经过 host queue、executor、future 或 callback 时，profiler 记录旧 execution context 到新 context 的 typed handoff edge。

到 CUDA 边界后，collector 把当前 causal token 绑定到 CUDA API invocation。CUPTI 提供 correlation ID 时，这个局部 ID 只是 graph 中的一条 edge，不再承担全局身份。Kernel、memcpy、memset、stream、context、graph 与 graph-node record 通过已验证的 CUPTI correlation path 继承 token。

闭源应用需要弱一些的路径。eBPF uprobe 或其他 userspace tracer 可以观察 CUDA runtime/driver 调用，并把它们关联到当前 process/thread work context。如果拿不到 framework-level token，profiler 就从 process-scoped root 开始，并把更高层 parent 标成 unknown，而不是猜一个请求身份。

Generation 用来避免 stale reuse。Stream handle、graph executable、context 与 framework object 都可能销毁后重新创建。Causal key 应该包含 object lifetime 或 generation，而不是假设一个 raw handle 永远全局唯一。

**与现有工作的差异。** CUPTI 已经有 API-to-GPU correlation 与 external correlation。这里新增的是更长、带 lifetime 的因果 namespace，把这些 ID 作为其中一层，并同时容纳 host task handoff 与显式 unknown root。

**可实现 artifact。** 一个 collector 加一套 portable trace schema，把 host scheduler/process evidence 与 CUPTI activity 连接起来。第一版可以复用仓库已有的 [xpu-perf](https://github.com/eunomia-bpf/xpu-perf) 与 CUPTI 示例，不需要新增 kernel interface。

**评测。** 构造 request 与 training-step workload，让工作在 launch 多 stream CUDA 之前刻意跨多个 host thread。随机化 thread pool，并重复复用 stream/graph object。对比 CUPTI-only 与 timestamp join，测 request-to-kernel parent-edge precision/recall、stale-ID collision、unknown-edge rate 与 overhead。

**学术价值。** 研究问题是：多个 runtime 的局部 identifier 与 lifetime 不共享 namespace 时，怎样保持跨 runtime 的 causal identity。

**生产价值。** Operator 可以直接问是哪一个 request、host task 或 process state 产生了问题 kernel，而不用手工拼接多个 dashboard。

**失败条件。** 如果 framework tracing 加 CUPTI correlation 已经能在真实异步 workload 上恢复几乎全部 parent edge，那么额外 token 层就是多余的，简单 local ID 应该胜出。

### 2. 带显式不确定性的 dependency-aware critical path

**缺口。** 即使 parent identity 正确，profiler 仍然需要解释 delay 是从哪里进入路径的：host enqueue、command-buffer submission、stream dependency wait、device execution，还是无关工作造成的 contention。

**机制。** 构建偏序 graph，而不是一张全局 timestamp 排序表。硬 edge 来自 same-stream order、CUDA event、CUDA Graph dependency、API-to-activity correlation 与已观测 host handoff。Interval observation 则包括 API start/end、可选 CUPTI queued/submitted timestamp、GPU start/end、synchronization wait、scheduler gap、page fault，以及相关 network/storage stall。

每条 edge 都带 evidence class：

- `explicit`：runtime 或 API 语义明确建立 dependency；
- `observed`：host handoff 或 system event 直接观测到关系；
- `inferred`：时间与状态支持这种解释，但没有证明；
- `missing`：建立这条关系需要的记录丢失或不可用。

Critical-path attribution 可以自动使用 `explicit` 与 `observed` edge。`inferred` 必须保留 confidence，不能偷偷升级成硬 dependency；`missing` 则阻止系统声称路径完整。

Profiler 之后可以把一个 late kernel 拆成多个 phase。API-to-queued 很长更像 host 或 driver preparation；queued-to-submitted 很长指向 command-buffer 或 driver delay；submitted-to-start 很长且存在 dependency predecessor 时，更像 stream/graph wait。若没有这种 predecessor，则可能是 device contention，需要进一步 device evidence。

**与现有工作的差异。** 现有 timeline 已经以不同形式展示 timestamp 与 dependency。这里的增量是一套显式 causal algebra，把 dependency proof 与 temporal coincidence 分开，并把 telemetry loss 直接带进 diagnosis result。

**可实现 artifact。** Graph builder 加查询接口：`why-late <kernel-id>` 返回 critical predecessor chain、每个 phase 的 delay、每条 edge 的 evidence class，以及第一个无法确定的边界。

**评测。** 每次只注入一个可控 delay：launch 前 CPU descheduling、host page fault、delayed memcpy、cross-stream event wait、graph dependency、device resource contention，以及无关 concurrent work。将返回的 critical path 与注入的 ground truth 对比，测 edge accuracy、root-cause top-1/top-3、false causal edge、unresolved-path recall 与 query latency。

**学术价值。** 可以验证带 evidence class 的偏序表示是否真的比 timestamp stitching 产生更可靠的因果解释。

**生产价值。** Profiler 能告诉工程师延迟是在哪一层进入 pipeline，以及下一步应该由哪一层负责调查。

**失败条件。** 如果 timestamp-only correlation 在 overlapping stream 和 confounder workload 上仍然达到相同因果准确率，那么 graph machinery 没有足够收益。

### 3. 专门制造歧义的 host-device causality benchmark

**缺口。** 单独报告 diagnosis accuracy 无法说明 causal explanation 是否正确，而普通 benchmark 往往让 root cause 在图上非常明显。

**机制。** 构建一个 generator，内部保存真实 dependency graph，同时刻意制造多个在时间上看起来都像原因的非原因。每个 test 生成一张 ground-truth work graph，包含 host task、CUDA API call、stream operation、graph node、memory copy、kernel、synchronization point 与可控 interference。

可以设计以下成对 case：

| Pair | 看起来相同的 symptom | 不同的真实原因 |
| --- | --- | --- |
| late launch vs device queueing | kernel 都晚启动 5 ms | host thread stall vs GPU-side wait |
| dependency vs overlap | 两个 kernel 时间上重叠 | 显式 event edge vs 独立 stream |
| slow copy vs blocked consumer | consumer kernel 晚启动 | PCIe transfer vs 无关 concurrent copy |
| missing record vs no edge | parent 看起来消失 | telemetry loss vs 本来就独立 |
| graph replay vs direct launch | 相同 kernel name 重复 | 复用 graph node vs 新 API invocation |

Runner 在已知位置注入 CPU scheduler delay、page fault、memory pressure、stream wait、PCIe contention、GPU occupancy pressure 与 record loss。Evaluation 直接计算 profiler 重建的 graph，而不是只看最终文字 label。

**与现有工作的差异。** 生产 profiling 系统已经证明跨层 diagnosis 可行。这个 benchmark 专门检查当 timestamp 被设计成具有误导性、local ID 被复用时，解释是否仍然正确。

**可实现 artifact。** 开源 workload generator、ground-truth graph format、trace corpus 与 evaluator。它应同时支持 CUPTI-only、host-only、combined trace 与 synthetic loss，让不同 profiler 可以在同样条件下比较。

**评测。** Baseline 包括 CUPTI-only correlation、host-only eBPF/perf tracing、nearest-timestamp stitching 和本文提出的 typed causal graph。主要指标是 parent-edge precision/recall、critical-path fidelity、root-cause accuracy、loss 情况下的 false-confidence rate 与 overhead，并分别报告 instrumented 与 closed-source mode。

**学术价值。** 把 causal fidelity 从“debug 成功案例”变成可测量性质。

**生产价值。** Tool builder 可以判断哪一层额外 tracing 真正值得其 overhead，也可以在 CUDA、driver 或 framework 升级后回归测试解释是否仍然可信。

**失败条件。** 如果简单 CUPTI correlation 加 timestamp 在这些 adversarial case 中仍能可靠恢复 ground-truth graph，那么 benchmark 会直接证明更强的跨层 instrumentation 没必要。

## eBPF 在这里很有用，但不应该独占整个设计

Host-side eBPF 可以在无需 application SDK instrumentation 的情况下观察 scheduler delay、process/thread transition、page fault、network activity，以及部分 userspace 或 driver boundary。此前的 [异步 eBPF causal profiler 报告](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/) 已经说明，即使没有 GPU，thread identity 也经常不足以表示逻辑工作。

GPU 又增加了一个拥有自己 dependency model 与 identifier 的 runtime。把所有 device relation 强行塞进 eBPF event schema 会丢掉 CUDA 已有的语义；只看 CUPTI 又看不到 CUDA runtime 外部的大量 host 原因。

更合理的架构是 federated：每一层保留自己最强的 native correlation，再用一个很小的 typed causal contract 连接它们。eBPF 是强 host evidence source，CUPTI 是强 CUDA evidence source，而 contract 应该允许其中任意一个 collector 被替换。

这也把本文问题与 device-side extensibility 分开。`gpu_ext` 说明纯 host-side eBPF 看不到所有关键 device event，并提出 GPU 内部的可编程 hook。这在原因发生于 kernel execution 内部时很重要。本文讨论的是另一个边界：在决定是否需要更深 device instrumentation 之前，怎样先保留 host work、异步 submission、dependency wait 与 device activity 之间可验证的关系。

## 第一版原型不需要新的 GPU runtime

第一版可以很克制：

1. 收集 CUPTI runtime/driver、kernel、memcpy、memset、synchronization、stream 与 graph activity；
2. 在支持的平台上启用 queued/submitted kernel timestamp；
3. 有应用或 framework boundary 时加入一个 host work token；
4. 用 uprobe 或其他 host tracer 收集 CUDA call 周围的 scheduler 与 system evidence；
5. 依据明确的 stream/event/graph 语义与 correlation ID 构建偏序 graph；
6. 注入已知的 host、dependency 与 device delay；
7. 对照 ground truth 计算 parent edge 与 critical path 的准确率。

只有这个实验真的发现不可观测的 causal boundary，才值得再增加 kernel、driver、framework 或 device hook。

这样可以保持研究问题本身清晰：缺少的不是“更多 GPU event”，而是**足够的 causal structure，让 profiler 能区分被证明的 dependency 与碰巧靠近的 event，并在证据不足时明确说无法完成解释。**

## 哪些结果会改变这个判断？

有三类结果会明显削弱新增 causal contract 的必要性。

第一，如果当前 CUPTI correlation ID、external correlation、CUDA Graph ID 与普通 framework tracing 已经可以在 multi-thread、multi-stream 与 graph-heavy workload 中高精度恢复 request-to-kernel critical path，那么新 namespace 基本是在重复现有机制。

第二，如果生产 diagnosis 实际只需要 cohort-level anomaly localization，像很多 differential debugging system 那样，那么精确到每个 operation 的 causal edge 可能不值得采集与分析成本。更便宜的跨层统计 profiler 会更合适。

第三，如果 benchmark 无法构造出 timestamp stitching 会给出错误因果解释，而显式 dependency tracking 能正确区分的 case，那么本文所描述的实践缺口比预期更小。

因此，下一步不是先做一个巨大的统一 GPU observability platform，而是先做 ground-truth workload，验证现有 local correlation 能否重建 causal graph。这个结果会直接决定更强的 host-device identity 与 dependency tracking 是否值得进入生产。

## 参考资料

- NVIDIA，[CUPTI Activity API usage and correlation](https://docs.nvidia.com/cupti/main/main.html)
- NVIDIA，[CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)
- NVIDIA，[CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- Linux kernel documentation，[Uprobe-based Event Tracing](https://docs.kernel.org/trace/uprobetracer.html)
- Zheng et al.，[SysOM-AI: Continuous Cross-Layer Performance Diagnosis for Production AI Training](https://arxiv.org/abs/2603.29235)
- Darzi et al.，[Host-Side Telemetry for Performance Diagnosis in Cloud and HPC GPU Infrastructure](https://arxiv.org/abs/2510.16946)
- Zheng et al.，[gpu_ext: Extensible OS Policies for GPUs via eBPF](https://arxiv.org/abs/2512.12615)
