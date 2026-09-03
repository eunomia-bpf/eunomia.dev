---
date: 2026-09-03
title: "eBPF 能看清 GPU Megakernel 里面到底在跑什么吗？"
description: "GPU Megakernel 把许多算子融合进一个持久内核，传统 kernel trace 会失去逻辑任务边界。本文讨论如何用带语义的 eBPF task hook 恢复可观测性。"
tags:
  - Daily Report
  - eBPF
  - GPU
  - Megakernel
  - Observability
  - Runtime Systems
research_question: "传统 kernel boundary 消失以后，GPU Megakernel 运行时需要什么接口，才能保留逻辑 task identity，并允许低开销、可后加载的 eBPF 观测程序安全地看到内部执行？"
source_cutoff: 2026-09-03
status: daily-report
---

# eBPF 能看清 GPU Megakernel 里面到底在跑什么吗？

一个 GPU 推理服务偶尔有几个请求特别慢，但 CUDA timeline 看起来几乎什么都没有：设备上只有一个长时间运行的 kernel。原本应该能看到的 attention、GEMM、通信和 sampling kernel 都不见了。

这些计算并没有消失。Megakernel 把它们变成了一个持久内核内部的 task，再由 GPU 上的 scheduler 自己分发。

这正是 Megakernel 想要的效果。它消除了 host launch gap 和粗粒度 kernel boundary，让更细的工作可以重叠执行。但同时，它也拿掉了很多 GPU profiler 最常用的一层解释边界。

本文的观点是：Megakernel 的可观测性不应该主要依靠事后从 PC 地址反推所有逻辑工作。**Compiler 和 in-kernel scheduler 本来就知道 task graph。更合理的做法是让它们暴露一小组带版本的 semantic hook，再让 device-side eBPF runtime 在不重新编译应用的情况下，把经过验证的 monitor 动态挂到这些 hook 上。** 这样，task latency、queueing、dependency stall 和 request attribution 可以先在事件附近聚合；CUPTI 与硬件 sampling 则继续提供 instruction-level evidence。

<!-- more -->

这不是另一篇“给 GPU 每条指令都插探针”的文章。之前的 [GPU instrumentation safety](https://eunomia.dev/zh/research/gpu-instrumentation-safety-contract/) 已经讨论过插入 observer 后怎样约束寄存器、控制流、资源开销和 coverage。之前的 [host/device causality](https://eunomia.dev/zh/research/gpu-host-device-causality/) 则讨论异步 CPU/GPU 事件如何建立因果联系。这里缺的是另一个对象：**当 compiler 主动消掉 kernel-per-operator boundary 以后，系统应该把什么当成可观测的执行单元？**

## Megakernel 主动消掉了传统 timeline 使用的执行边界

传统 GPU 执行模型天然给 profiler 提供了一些结构。Host thread 往 stream 里提交 kernel 和 memory operation，trace 可以记录 launch、kernel instance、开始和结束时间、stream，以及周围的 CUDA API。NVIDIA 当前的 [CUPTI 文档](https://docs.nvidia.com/cupti/) 也把 Activity API 用于 CUDA API、kernel、memory operation 的 tracing，同时提供 PC Sampling、SASS metric 等更底层的信息。

Megakernel 恰好是在主动改变这个结构。

OSDI 2026 的 [MPK](https://www.usenix.org/conference/osdi26/presentation/cheng) 会把多 GPU 模型推理变成一个 persistent megakernel。Compiler 把 tensor program 降成 SM 粒度的 task graph，in-kernel runtime 再把这些 task 分发到不同 SM。论文评测中，相比传统 kernel-per-operator 的 serving baseline，端到端 latency 最多降低 1.7 倍。对可观测性来说，更重要的不是这个最大数字，而是执行结构：大量逻辑算子和通信步骤现在都处于同一个 kernel instance 里面。

[Event Tensor](https://arxiv.org/abs/2604.13327) 又把这种结构推广到动态 workload。它把 tiled task 之间的完成依赖表示成一等的 Event Tensor，支持 shape-dependent 和 data-dependent task graph，再把它们降成使用静态或动态 GPU 内部调度的 persistent kernel。比如 MoE 中，运行时 routing result 可以决定哪些 tile 去更新哪些 event，以及哪些后继 task 被触发。

这里出现了一个很有意思的反转：

- Compiler 内部反而拥有比以前更多的语义结构，因为它显式处理 task、event、dependency、symbolic shape 和 scheduling transformation；
- 外部 kernel timeline 反而拥有更少结构，因为这些工作可能全部只显示成一个 persistent kernel。

所以逻辑边界并不是彻底消失了。它只是从 CUDA launch interface 移到了 compiler IR 和 in-kernel scheduler 里面。

## PC Sampling 能告诉你哪段指令热，却不能自动告诉你这是哪个逻辑 task

CUPTI 的 PC Sampling 可以采样 warp program counter 和 scheduler state，也能给出 stall reason。这在 persistent kernel 里依然很有价值。如果某一段 instruction 长时间受 memory 或 execution dependency 阻塞，PC sampling 可以指出那一段代码。

但 PC identity 和 application task identity 回答的是两类问题。

同一个 device function 可能服务很多 request、decode step、expert、tile 或 communication epoch。Dynamic scheduler 也可能在不同 dependency 状态下，把相同代码分发给不同 SM。因此两个落在同一个 PC 的 sample，可能属于完全不同的逻辑 task，背后的慢因也不同。反过来，一个逻辑 operator 也可能被拆成很多 task type 和很多 PC region。

当然可以用 debug metadata、compiler IR、binary address 和 runtime state 再把语义拼回来。如果 profiler 和 compiler 本来就是一套系统，这很合理。但如果目标是生产环境里后加载一个新查询，它就不是很好的通用 contract。一个地址并不等于下面这种稳定语义：

```text
request_generation = 1842
decode_step = 37
logical_op = attention_qk
tile = (head=12, block=6)
task_generation = 991733
wait_reason = dependency_event_481
```

真正缺的不是另一种 stack trace，而是一个稳定方法，让 observer 可以直接问 in-kernel scheduler：“你现在执行的是哪一份逻辑工作？”

## 现有 Megakernel runtime 已经证明 task-level profiling 本身并不难做到

一个非常强的反对意见是：根本不用 eBPF，每个 Megakernel compiler 自己实现 profiler 就行。

这不是理论上的替代方案。[Mirage MPK 仓库](https://github.com/mirage-project/mirage/tree/mpk) 已经提供 `--profiling` 模式，可以显示每个 task 的 execution timeline；它的 persistent-kernel API 里也有 `profiler_tensor`。MPK 本来就掌握 task graph 和 scheduler state，因此 compiler-owned profiler 可以直接记录有意义的内部事件，不需要从外面的 kernel boundary 猜。

所以不能为了“可编程”三个字就再叠一层 runtime。新的接口只有在提供了 compiler-native profiler 明显不具备的能力时才值得存在。

这里最有价值的差异是 **late-bound programmability**。生产事故里的问题往往是部署之后才知道的：只看某一个 request generation 的 task；统计 dependency wait 超过 20 微秒的 task；把一个 expert routing imbalance 和某个 communication phase 对上；或者只有 SLO 已经危险时才提升采样率。为了这种临时问题重新编译 Megakernel，或者一开始就打开完整 task trace，代价可能都太高。

[gpu_ext](https://arxiv.org/abs/2512.12615) 给出了一个可能的执行机制。它在 GPU driver 暴露 hook，并引入可以在 GPU kernel 内执行 verified policy logic 的 device-side eBPF runtime。但这并不会自动解决 Megakernel observability。Verifier 可以限制 monitor 的行为，却无法凭空制造 compiler 没有暴露的 task semantics。

因此真正有意思的组合是：

1. compiler/runtime 暴露 semantic scheduler hook；
2. device-side eBPF runtime 在这些 hook 上提供安全的 late-bound program；
3. monitor 在设备侧先做 bounded aggregation，只把 summary 或命中的少量 evidence 送回 host，而不是把每一个 task event 都导出。

在这个设计里，eBPF 是核心执行机制，不是顺便用一下的 host tracer。

## Semantic hook 应该比 compiler IR 小，也应该比 PC 丰富

把完整 Megakernel IR 暴露给每个 profiler，会让工具和 compiler implementation 强耦合。只暴露 raw PC 又会把逻辑语义丢掉。合适的接口应该处在两者中间。

第一版 semantic hook ABI 可以只覆盖少数 scheduler event：

- task 变成 ready；
- task 被分配给 worker 或 SM；
- task 开始和完成；
- task 等待或释放 dependency event；
- task 参加某个 communication operation；
- request 或 decode-step generation 前进；
- scheduler queue 超过一个声明过的 pressure threshold。

Hook context 只暴露稳定 ID 和 bounded metadata，而不是任意 compiler object。比如 task-class ID 可以查一张 side table，得到它对应的 source operator、生成后的 device function、tensor region class 和 dependency class。Request-generation token 用来区分复用的 request slot。Task-generation number 用来区分同一个逻辑 tile 的重复执行。

ABI 还必须定义 capability boundary。Monitoring program 可以读 task metadata、更新自己的 map，但不能改 scheduler state。Scheduling policy 如果以后需要支持，可以使用另一种 program type，只允许返回非常窄的 priority hint。不要在一个叫“observability”的 hook 里偷偷塞入控制能力。

这正是 eBPF-like model 比较合适的地方。Program type、attach point、context schema、map visibility、helper set 和 verifier rule 可以一起定义一个比 compiler implementation 小、又比 program counter 丰富的 contract。

## 目前仍然薄弱的地方

### Kernel-level correlation 在 persistent kernel 内失去了真正的分母

现有 GPU tracing 可以关联 host API、kernel instance、stream、source location、PC 和 hardware sample，这些信息仍然不可缺少。但一个 persistent megakernel 里面可以包含成千上万甚至更多 logical task execution。

如果没有 task denominator，“35% 的 sample 落在这段 instruction”并不能告诉我们：这个 task class 普遍都慢，还是只有一个 request generation 异常；是执行本身慢，还是 scheduler 反复饿死了一小部分已经 ready 的 task。

一个有区分力的实验应该保持 Megakernel binary 完全相同，只改变内部 task schedule 或 request mapping。如果 profiler 对两个 ground truth 不同的 bottleneck 给出同一个解释，那么 kernel/PC identity 对这个问题就不够。

### Compiler-native profiler 有语义，但还不是跨 runtime 的 late-bound interface

MPK 已经证明 compiler 可以输出 task timeline，Event Tensor 也把 task 与 dependency event 放进一等 compiler representation。这些应该作为最强 baseline，而不是假装不存在。

剩下的问题是：operator 能不能在部署之后加载一个很小的新查询，而不重新编译 Megakernel、不打开完整 trace，也不要求外部 profiler 理解每个 compiler 的内部格式。当前还没有一个跨 Megakernel runtime 的 task-observability contract。

最简单的答案也可能只是一种统一 export format，而不是 eBPF。任何 eBPF 方案都必须在 query flexibility、overhead 或 deployment safety 上明显赢过这个更简单的 baseline。

### Dynamic task graph 需要 generation 和 coverage，而不只是静态 operator name

在 dynamic MoE 或 continuous batching 中，task 是否存在、依赖谁，都可能由运行时数据决定。只记录一个静态 operator label 不够。一个可信 record 需要能区分 task generation，也要记录让它变成 runnable 的 dependency state。

Coverage 同样是动态的。Monitor 可能只 sample 某个 task class，压力大时 throttle，或者 telemetry budget 用完后跳过事件。查询结果必须携带 eligible-event denominator 和 loss state。否则一个很低的计数既可能代表“这件事很少发生”，也可能代表“observer 几乎没看到它”。

本文直接沿用之前 [instrumentation-safety contract](https://eunomia.dev/zh/research/gpu-instrumentation-safety-contract/) 对资源扰动的约束。Megakernel observability 新增的是另一条要求：面对不断变化的 task graph，如何说明 semantic coverage。

### 跨 GPU dependency 很容易被误判成 local scheduler 问题

MPK 会把多 GPU computation 和 communication 一起放进 Megakernel。一个本地 task 可能只是因为远端还没有产出对应数据或 event 而等待。如果 device-local profiler 只记录本地 worker 状态，就可能把这种等待误判成 local scheduler imbalance。

所以 semantic interface 至少需要一点 distributed identity，把 task generation 和 communication/remote dependency generation 联系起来，但又不能为了每个 tile 都导出完整 distributed trace。这个 evidence budget 现在还没有明显正确答案。

## 同时具有学术和生产价值的方向

### 1. 为 Megakernel scheduler 定义 versioned semantic hook ABI

**缺口。** Megakernel compiler 知道 task identity 和 dependency，但外部工具要么只看到一个 kernel，要么必须依赖 compiler-specific profiling format。

**机制。** Compiler 在 lowering 时生成一份紧凑的 task schema，并给 scheduler event 生成稳定 hook descriptor。每个 hook context 只暴露 bounded field，例如 task class、task generation、request/decode generation、dependency class、worker/SM identity 和 communication generation。新的 device-side eBPF program type 可以挂到这些 hook 上，只读取声明过的字段。Compiler 内部 IR 可以继续演化，只需要显式 version 或维持外部 ABI。

这个 ABI 应该明确分开 observation hook 与 control hook。Observability program 可以聚合并输出证据，但不能修改 scheduler state。未来如果需要可编程调度，应使用另一种 program type，给它更窄的 return contract。

**相对已有工作的增量。** MPK profiler 已经利用 compiler-owned task semantics；gpu_ext 已经展示 verified device-side eBPF execution。这里的新东西是两层之间的 contract：给动态加载 monitor 一个可移植的 semantic attach surface，而不是再造一个 Megakernel compiler 或 binary instrumentation framework。

**产物。** 在 MPK 或 Event Tensor backend 中导出 task schema，在选定 scheduler transition 上调用 gpu_ext-style eBPF hook，再提供一个 host loader，可以在不重建 model engine 的情况下 attach 或替换 monitor。

**评测。** 对 static 和 dynamic task graph 测量 hook cost、register/resource delta、attach latency、task coverage 和 diagnosis accuracy。基线至少包括 compiler-native profiling、PC sampling、always-on task logging，以及相同 telemetry budget 下的 eBPF hook。

**学术价值。** 核心问题是：传统 kernel boundary 消失后，由 compiler 创造出来的 execution semantics 能不能成为稳定的 systems ABI。

**生产价值。** Operator 可以直接对正在运行的 inference service 提一个新的定向问题，而不是部署一个重编译 engine 或打开高开销 universal trace。

**失败条件。** 如果 task schema 变化太快，稳定 ABI 要么泄漏完整 compiler IR，要么丢掉诊断所需信息，那么跨 runtime semantic hook 就不是合适抽象。

### 2. 在 Megakernel 内做携带 coverage 的 task aggregation

**缺口。** 把每个 task transition 都导出，会重新引入 Megakernel 原本想消除的 telemetry 与 synchronization 成本；只做 sampling 又可能因为没有 denominator 而给出很自信的错误 summary。

**机制。** 第一阶段聚合直接在 device-side eBPF monitor 中完成。按 task class 或 request generation 维护 bounded counter/sketch，记录 ready-to-start delay、execution time、dependency-wait class、queue depth，以及少量 hardware sample correlation。每个 aggregation epoch 同时记录 eligible event、observed event、throttled event、lost record 和 monitor/program generation。

Host 先收 compact summary，只有 predicate 命中后才升级到少量 raw event。比如“哪个 task class 解释了 p99 decode-step gap”可以先用 semantic aggregate 定位，再只对那个范围抓更细的 evidence。

**相对已有工作的增量。** CUPTI 提供 kernel、PC 和 hardware evidence，compiler-native profiler 可以给 task timeline。这里新增的是在内部 scheduler boundary 上做可编程 semantic aggregation，并把 observation coverage 一起作为结果，而不是增加一种固定 trace format。

**产物。** 一组 verified task monitor，加上一种 coverage-aware result format，可以和 CUPTI PC/PM sample 关联，同时明确承认并不是每一个 hardware sample 都有可靠 logical task identity。

**评测。** 从小模型到高度 tiled 的 MoE workload 逐步提高 task-event rate，固定总 telemetry bandwidth，比较 raw task tracing、fixed-rate sampling、compiler-native profiling 与 coverage-carrying eBPF aggregation 的 root-cause accuracy、输出字节数、device overhead 和 false-confidence rate。

**学术价值。** 这可以检验 semantic observability 是否能被建模成显式的 on-device evidence budget，而不是简单的 trace 开关。

**生产价值。** Always-on monitor 可以保持低成本，同时在罕见问题出现时仍有升级到深层证据的路径。

**失败条件。** 如果 compiler 维护少量固定 counter 就能在不同 workload 下达到同样诊断质量，可编程 aggregation 只是多余复杂度。

### 3. 建立“kernel fusion 后还能不能诊断”的 counterexample benchmark

**缺口。** Megakernel 的评测通常关心 fusion 后快了多少，却很少构造这样的 paired case：外层 persistent kernel timeline 几乎一样，但内部真正 bottleneck 不同。

**机制。** 从同一个 task graph 和同一个 binary 出发，在 task scheduler 注入可控故障：延迟某个 task class、制造 MoE routing skew、推迟一个 dependency notification、压满一个 worker queue、延迟一个 communication generation，或者反复饿死一个已经 ready 的 tile。尽量保持外层 kernel launch 和总 runtime 接近，但让内部 root cause 已知且不同。

Ground truth 由 compiler task graph 和注入的 scheduler event 给出。所有工具使用同样的 overhead 或 telemetry budget，并且必须指出受影响的 logical task class 和 cause。再加入 conventional kernel-per-operator build 和 CUDA Graph build，用来验证在哪些情况下普通 kernel boundary 本来就已经够用。

**相对已有工作的增量。** 这不是 Megakernel speed benchmark，而是把 fusion 本身当成一次 observability transformation，检查 kernel-level 或 PC-level evidence 在什么条件下已经不能保持正确诊断。

**产物。** 一个 CUDA-first benchmark，包含 MPK/Event Tensor 风格的 task graph、fault injection、已知 task/dependency identity，以及 CUPTI、compiler-native profiling 和 semantic eBPF hook adapter。

**评测。** 主要指标是 cause-identification accuracy、task-attribution accuracy、false confidence、telemetry bytes、runtime perturbation，以及 attach 一个新 query 的时间。最强 baseline 必须是 compiler 自己的 profiler，而不是故意挑一个弱 trace。

**学术价值。** 它把“Megakernel 更难观测”从一句经验判断变成可证伪的系统性质，并且能指出到底是哪一层 semantic boundary 真正重要。

**生产价值。** Compiler 与 observability 团队可以在引入新 fusion strategy 时做 regression test，避免性能变快的同时悄悄把事故诊断能力删掉。

**失败条件。** 如果相同预算下，CUPTI PC sampling 加普通 compiler metadata 就能和 semantic hook 一样可靠地找出注入的 task-level cause，那么额外 eBPF 层没有必要。

## 哪些结果会改变这个判断？

有三类结果会明显削弱 semantic eBPF hook 的必要性。

第一，Megakernel compiler 可能很快收敛出足够便宜、支持动态过滤、并且 production-stable 的内建 task profiler。MPK 已经证明 task-level profiling 完全可以由 runtime 原生提供。如果这些 profiler 还能接受 late-bound predicate，并明确给出 coverage，那么再加一层 programmable monitor 的收益就很小。

第二，硬件和 vendor tooling 可能提供更丰富的 in-kernel semantic range。CUPTI 已经有 kernel tracing、PC Sampling、SASS metric 和 range profiling。如果未来硬件几乎零成本地把这些 sample 直接关联到 compiler-defined task ID，更合适的标准可能是 hardware/compiler metadata channel，而不是 eBPF。

第三，counterexample benchmark 可能证明 PC 加 compiler metadata 已经够。只要同一个 binary 里的不同内部 bottleneck 不依赖 scheduler hook 也能稳定诊断，再引入一个新的 device runtime boundary 就只是在增加系统复杂度。

目前证据更支持一个较窄的结论：**Megakernel observability 应该跟着 compiler 新创造出来的 semantic task boundary，而不是继续依赖它刻意删除的 kernel boundary。** Device-side eBPF runtime 的价值在于把这个边界变成 late-bound、可编程的接口，但它只有在严格 overhead 与 coverage budget 下，确实比 compiler-native profiling 更能回答真实诊断问题时才值得存在。