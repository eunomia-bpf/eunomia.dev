---
date: 2026-08-29
title: "GPU 运行时只看页故障，能把内存放对地方吗？"
description: "GPU 显存超配会把页面放置变成持续变化的运行时决策。本文分析页故障、访问采样、对象语义和调度信息各能证明什么，并提出可验证的内存放置证据契约。"
tags:
  - Daily Report
  - GPU
  - Unified Memory
  - Memory Management
  - Runtime
research_question: "GPU 运行时需要暴露并保留哪些证据，才能让显存超配下的迁移、预取、驱逐和远程访问决策可解释、可比较并可验证？"
source_cutoff: 2026-08-29
status: daily-report
---

# GPU 运行时只看页故障，能把内存放对地方吗？

假设两个 GPU 任务合起来会访问 120 GB 数据，而 GPU 只有 80 GB HBM。Unified Memory 让两个任务继续使用统一虚拟地址，不要求应用自己把所有数据切成 host 和 device 两套；代价是总有一部分页面必须留在 host DRAM。某个 kernel 最终访问了一个不在 HBM 的页面，于是 GPU 触发页故障，运行时还得先腾出空间。

这个 fault 只能确定一件事：当前确实需要这个页面，而且它此刻不在这里。它回答不了后面的决策。应该驱逐哪一个 HBM 页面？新页面值得真的迁进来，还是让 GPU 远程访问更划算？被赶走的页面会不会 200 微秒后马上又被用到？当前访问只是一次短暂 phase，还是一个会被反复读取的 tensor？正在运行的任务是不是马上就会被切走？

这些信息决定了 oversubscription 最后只是多几次传输，还是演变成持续的 page thrashing。近两年的 GPU memory systems 也越来越少只靠 fault 做判断：有的增加 HBM access sampling，有的利用 compiler 推断出的对象访问语义，有的跟踪 object/phase，还有的直接把 GPU scheduling timeline 交给 memory manager。真正缺的已经不只是“怎么移动一个页面”，而是“运行时凭什么认为这次移动是对的”。

<!-- more -->

本文提出一个 **携带证据的 GPU 内存放置契约（evidence-carrying GPU memory placement contract）**：每次 migration、prefetch、eviction、replication 或 remote access 决策，不只留下动作结果，还保留触发它的观测、这些观测有多新、覆盖了多少，以及它正在满足什么 application 或 scheduler intent。这样不同 memory policy 才能在同一组证据语义上比较，而不是各自把假设藏在 driver 里。

这是一篇 adjacent-systems 报告，不计作 eBPF-centered。eBPF 或类似的可编程 instrumentation 可以补充 host/runtime 侧证据，[bpftime 的 GPU 工作](https://eunomia.dev/bpftime/documents/gpu/)也能作为实验入口，但这里的核心问题并不依赖 eBPF。

## Unified Memory 统一了地址，但没有固定页面住在哪里

CUDA Unified Memory 让 CPU 和 GPU 可以访问同一份 allocation，具体 backing page 由 runtime 决定放在 host 还是 device。当前 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) 对 `cudaMemAdviseSetPreferredLocation` 的表述很明确：它是 performance hint，不是 residency guarantee。`cudaMemPrefetchAsync` 可以请求把某段数据提前迁到指定 processor 附近，之后新的访问或其他 hint 仍然可能把它移走。

当前 [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) 还有一个很值得注意的细节。默认情况下，driver 检测到 host/device 之间持续 thrashing 时，可能最终把页面固定在 host memory；但如果应用把 GPU 设成 preferred location，这个 preference 可以覆盖原来的 thrash resolution，让页面继续来回迁移。也就是说，应用表达的是“倾向”，这个倾向会改变 policy，却没有形成一个“运行时必须满足并反馈结果”的 placement contract。

Linux HMM 的底层也同样是动态的。当前 [Linux HMM 文档](https://docs.kernel.org/mm/hmm.html) 通过 `migrate_vma_*()` 支持 system memory 与 device-private memory 之间迁移。driver 可以决定一个 range 里的哪些 page 实际迁移，还要处理 race、无法迁移的页面以及 device mapping 更新。虚拟地址保持有效，并不意味着这些地址背后的 physical placement 不会变化。

AMD 的 [HIP managed memory](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html) 也提供类似模型。在支持 HMM 和 recoverable page fault 的系统上，页面可以自动在 host 与 device 之间迁移；具体能力取决于 GPU architecture、kernel support 与 XNACK 配置。

这些接口的共同点是：统一 addressability 相对容易，稳定 residency 并不是它们承诺的抽象。应用可以一直拿着同一个 pointer，底层却不断重新做 placement decision。

## 最近的系统为什么都在给 memory manager 增加新证据

Page fault 很适合证明“现在有人要这个页面”，却几乎不包含“接下来谁更值得留在 HBM”这种 future reuse 信息。

ISCA 2026 的 [Observability-aided GPU Memory Oversubscription](https://www.csa.iisc.ac.in/~arkapravab/papers/ISCA26_ObservUVM.pdf) 直接展示了这个缺口。NVIDIA UVM 的 CPU-side driver 能看到访问 DRAM-resident page 产生的 fault，但一个 page 已经迁入 HBM 后，driver 通常没有直接办法继续观察 GPU 对它的访问。因此默认 eviction 顺序更接近“最近何时迁入”，而不是“最近是否真的被使用”，活跃页面也可能被赶走。论文把原本用于 PCIe access tracking 的 hardware access counter 改造成 HBM-resident region 的采样观测，并构建 ObservUVM，把 eviction/prefetch policy 放到 userspace。对 14 个应用的评估中，这组 observability-aided policy 相对 UVM baseline 报告了 34% 的 geometric-mean speedup。

这里最值得记住的并不是 34% 本身，而是额外观测会改变“该赶走谁”的答案。单靠 fault stream 根本没有这部分信息。

其他系统补的是不同层次的证据。MICRO 2024 的 [SUV](https://www.csa.iisc.ac.in/~arkapravab/papers/MICRO24_SUV.pdf) 把 compiler 推断出的 access semantics 与 runtime 信息结合起来，判断哪些 object 或 object region 更值得占用有限 HBM，以及何时应该主动 prefetch。HPCA 2025 的 [OASIS](https://yueqiwang42.github.io/assets/pdf/papers/OASIS_HPCA25.pdf) 进一步发现，同一个 multi-GPU 应用里，不同 object、甚至同一个 object 的不同 phase，最合适的 page-management policy 都可能不同，因此它把对象与 phase 行为带入 migration/duplication 的选择。

HPCA 2026 的 [ARIADNE](https://doi.org/10.1109/HPCA68181.2026.11408564) 仍然保持 UVM 对应用透明的抽象，但在 runtime 里计算 Sharing Degree，pipeline fault handling，并根据实时 access pattern 在 GPU memory 与 zero-copy 之间动态放置 region。它的 [artifact](https://zenodo.org/records/17830000) 也公开了 driver-level 实现。

[MSched](https://arxiv.org/abs/2512.24637) 又走得更远。它不把每个 fault 当成独立的意外，而是让 GPU scheduler 和 memory manager 协同。kernel launch 信息加上已知的 scheduling timeline 可以预测未来 working set，在 task context switch 前主动准备内存。论文在其 multi-task oversubscription workload 上报告了远高于 demand paging 的性能。

这些系统对“最好的证据是什么”没有统一答案，但对问题本身已经很一致：fault 只描述眼前 demand。要把页面放得更好，运行时还得知道 resident page 是否活跃、object 是什么、程序处于哪个 phase，或者接下来哪个 task 会运行。

站内此前的 [page-level eBPF memory attribution](https://eunomia.dev/research/page-level-ebpf-memory-attribution/) 关注的是另一件事：怎样把 page-level cost 重新连回造成它的 application object。即使这个 attribution 已经完全正确，GPU memory manager 仍然要回答下一步哪个页面更应该留在 HBM。

## 更多 observability 本身也有成本

“多收一点信息”并不天然正确。ObservUVM 之所以做 sampling，就是因为在 HBM 的极高带宽下完整跟踪每一次访问不现实。NVIDIA [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 可以采集 Unified Memory CPU/GPU page fault，但文档明确提醒，相关 fault tracing 在测试中可能带来最高 70% 的 overhead。

Static information 有另一类失效方式。compiler 能理解的代码可以给出很精确的 pattern，但 closed-source library、data-dependent indexing 和 runtime phase change 都可能让预测失真。Object-aware policy 需要跨 allocator/library 稳定识别对象。Scheduler knowledge 在受控 multi-task runtime 里很强，但面对来自独立 process 或外部 service 的 work 时未必完整。

所以 production design 不能简单变成“把所有信号都收齐”。真正需要的是让 runtime 知道每一种 evidence 能支持什么结论，以及它什么时候已经 stale、只覆盖 sample、缺失，或者与另一种证据矛盾。

## 现有研究还缺什么

### 不同 placement evidence 没有共同语义

GPU fault 表示某个 address 在 non-resident 状态下被 demand。Access-counter notification 表示一个 sampled region 的 activity 达到了阈值。Static analysis 根据 code structure 预测未来访问。Object-aware system 把行为绑定到 logical allocation。Task scheduler 则可能直接知道一个 process 什么时候会再次拿到 GPU。

这些信息最后都可能影响同一次 eviction，却没有统一的 evidence model。结果是 policy 很难横向比较，也很难组合，除非把某个系统的特定假设直接写进另一个系统。

一个有区分度的实验是：给几种 policy 完全相同的 workload 与 fixed observability budget，然后要求每次 migration 都只能引用“决策发生前已经可见”的证据。如果两个 policy 最终 throughput 一样，但其中一个经常使用过期或互相冲突的 evidence，这个差异应该直接被 benchmark 看见。

### Placement hint 没有告诉上层“意图到底有没有被满足”

CUDA 把 memory advice 定义成 hint 是合理的，它让 driver 保留实现自由。但 production control plane 因此缺了一层。应用可以请求 preferred location，也可以发起 prefetch，却没有一个 portable service-level object 能表达：“这 8 GB region 在这次 inference phase 内优先留在 GPU；除非 memory pressure 超过 X，否则不要驱逐；如果做不到，请明确告诉我原因。”

这里不是要把每个 `cudaMemAdvise` 都升级成硬保证，而是需要一个可选的上层 contract，把 desired placement、允许怎样降级，以及 runtime 实际满足到什么程度记录下来。

测试可以用两类数据做：一类对 tail latency 很敏感，一类可以安全 spill。在固定 memory pressure 下，contract-aware runtime 应该保持这个优先级，或者明确报告无法满足；hint-only baseline 则可能静默地破坏它。

### 现在的 evaluation 很少直接测“这次 decision 做得对不对”

GPU memory paper 通常比较 application runtime、page-fault count、migration traffic 或 throughput，这些指标都合理，但不能说明 policy 是不是因为正确证据做出了正确决定。

一个 policy 可能刚好撞对了某种 access pattern；也可能通过 aggressive prefetch 降低 fault，却把 PCIe bandwidth 用光，在共享链路上伤害另一个 workload。更强的 benchmark 需要知道 future reuse 的 ground truth，还要固定 evidence budget，才能测 decision regret，而不是只看最终 runtime。

### 跨 vendor 的 portability 到 policy 这一层就变得模糊

CUDA UVM、HIP managed memory 和 Linux HMM 都支持某种 shared addressing 与 migration，但它们暴露的 hardware signal、fault 行为、coherence mode 和 policy hook 并不相同。Portable runtime 不能只是把 vendor event 换个名字，就假设语义相同。

因此更合适的抽象不是“全平台统一 eviction algorithm”，而是一小组稳定概念：evidence、placement intent、action 和 uncertainty。具体 adapter 应该保留 vendor-specific fact，而不是把差异抹掉。

## 兼具学术价值与生产价值的方向

### 1. 让每一次 placement decision 自带证据

**Gap。** 现在的 memory manager 可以利用 fault、access sampling、object semantics 或 scheduling prediction，但某一次 placement action 为什么发生，通常被藏在具体 driver/policy 代码里。

**Mechanism。** 给每个 managed virtual region 分配 lifetime-scoped region ID 和 generation。只要 runtime 改变 placement，就输出一个紧凑 decision record：

```text
region_generation
virtual_range
action = migrate | evict | prefetch | remote-map | replicate | keep
evidence = [fault, sampled_access, object_phase, kernel_prediction, schedule]
evidence_age
coverage_or_sampling_rate
pressure_state
policy_generation
confidence
```

这里不把不同 evidence 当成同等强度。Fault 可以准确证明一次 access；sampled access 只给概率性 hotness；scheduler prediction 有过期时间；compiler claim 可能识别 region，却漏掉 data-dependent access。Schema 需要保留这些差异。

高频路径不必保存每条完整日志。driver 可以按 policy generation 聚合，只保留异常、高成本或会改变 eviction outcome 的 exemplar。这样 operator 可以回答“为什么这个 page 被移走”，又不用长期打开 full memory-access tracing。

**与已有工作的差别。** ObservUVM 已经把 mechanism 和 policy 分开，并提供 sampled access evidence；SUV、OASIS、MSched 分别增加更高层语义。这里要标准化的是 evidence-to-decision boundary，而不是再发明一个新的 eviction heuristic。

**Artifact。** 一个 trace schema、CUDA/HIP/HMM adapter，以及 `why-moved <region>` 查询工具。现有 [bpftime GPU runtime](https://github.com/eunomia-bpf/bpftime) 可以用它做 instrumentation 实验，但 schema 本身不依赖 bpftime。

**Evaluation。** 在 oversubscribed scientific kernel、CUDA Graph workload、DNN/LLM inference 和 mixed multi-process workload 上比较 fault-only、sampled-access、compiler/object-informed 与 schedule-aware policy。测量 record overhead、无法解释的 decision 比例、stale-evidence rate，以及 postmortem query 能否还原真实 migration 原因。

**Academic value。** 把 GPU memory observability 与 placement policy 之间原本隐式的边界变成可比较的系统对象。

**Production value。** Operator 可以区分“应用就是超过了 HBM”与“policy 把仍然 hot 的 region 赶走了”，而不必采集所有 memory access。

**Failure condition。** 如果 decision record 对 diagnosis 和 policy comparison 都没有比现有 fault/migration trace 多提供有效信息，就不值得保留这层 metadata。

### 2. 把 placement intent 与实际 compliance 一起暴露出来

**Gap。** 现有 advice API 能表达 preference，却很难让更高层 scheduler 知道 runtime 是否真的满足了某个 phase 的内存优先级。

**Mechanism。** 在 vendor-specific advice 上增加一个可选 runtime object。一个 region generation 可以声明：

- 希望驻留在哪些位置，例如 GPU 0 HBM 或 host DRAM；
- 哪些 access mode 可以 remote-map，而不必 migration；
- migration deadline 或 phase boundary；
- 相对其他 region generation 的 eviction priority；
- 最大允许 remote-access 或 migration budget；
- 与 kernel、graph、request 或 scheduler epoch 绑定的 expiry condition。

Memory manager 再把这些 intent 映射到 CUDA advice、HIP advice、HMM migration 或自己的 policy。如果因为 capacity、topology、coherence 或更高优先级 region 无法满足，就记录 degraded state 与原因，而不是把 hint 发出去后默认它已经成功。

Generation 很重要。同一块 tensor buffer 或 virtual range 可能在下一阶段被复用，旧的 placement intent 应该随 logical lifetime 失效，而不是永远黏在一个 address 上。

**与已有工作的差别。** CUDA/HIP 已经有 placement/prefetch hint；MSched 引入 scheduler knowledge；OASIS 把 policy 绑定 object 与 phase。这里新增的是跨这些机制都能表达的 requested outcome 与 observed compliance。

**Artifact。** 先基于 NVIDIA UVM 实现 userspace controller 和小型 region-intent API，再做 HIP/HMM compatibility prototype。controller 同时保留普通 best-effort hint 和“做不到就报告 degradation”的模式，不要求现有硬件突然提供新的 correctness guarantee。

**Evaluation。** 构造不同 latency sensitivity 的 paired region，在 110%、150% 和 250% HBM subscription 下运行，并加入 multi-GPU peer access、CPU/GPU ping-pong、phase change 与 concurrent jobs。与 plain UVM 和 static advice 比较 intent satisfaction、migration bytes、remote-access bytes、tail kernel delay、throughput 与 control overhead。

**Academic value。** 研究 memory placement 能不能从 fault 的隐式副作用提升为显式 resource contract。

**Production value。** Inference/training runtime 可以表达 KV-cache working set、communication buffer 或下一批 imminent data 比 cold model state 更值得占 HBM，同时在压力太大时拿到真实失败原因。

**Failure condition。** 如果 static advice 在 phase-changing 与 multi-tenant workload 上也能达到同样的 tail latency 和 migration cost，这层 contract 就没有必要。

### 3. 用 counterexample benchmark 直接测试 placement evidence 的价值

**Gap。** 现有 evaluation 能看出最终性能差异，却很少刻意制造“表面观测相同、正确 decision 不同”的情况。

**Mechanism。** 构造 paired workload，保持一种可见 signal 近似不变，同时改变真正最优的 placement：

| Pair | 保持相似的观测 | 改变最优动作的隐藏事实 |
| --- | --- | --- |
| repeated faults | fault stream | 一个页面马上会复用，另一个已经 dead |
| equal access counts | sampled hotness | 一个 region 的访问即将发生，另一个很久以后才用 |
| same allocation size | object metadata | 一个 phase 只访问 5%，另一个访问 95% |
| same current residency | HBM state | 一个 task 马上被 schedule out |
| same preferred location | advice | 一个 job 有严格 latency budget，另一个可以 spill |

Harness 自己知道 future reference 与 task order，因此能在同样的 HBM/PCIe 约束下计算 oracle placement。然后逐层开放 evidence：fault-only、sampled access、object/phase semantics，最后才给 scheduler knowledge。

**与已有工作的差别。** ARIADNE、ObservUVM、SUV、OASIS、MSched 已经证明了不同 policy mechanism 可以工作。这个 benchmark 不再只比较谁跑得快，而是隔离“每增加一种 evidence，到底减少了多少错误 decision”，并故意让隐藏假设暴露出来。

**Artifact。** 开源 trace corpus、可 replay 的 oversubscription workload、oracle solver 与 evaluator。第一版支持 NVIDIA UVM，同时保持 vendor-neutral trace format，后续可以加入 HIP/HMM。

**Evaluation。** 核心指标包括 application runtime、fault stall time、migration/remote-access bytes、useful-prefetch ratio、相对 oracle 的 decision regret、false-confidence rate 和 observability overhead。每种 policy 都必须拿到完全相同的 memory、link 与 evidence budget。

**Academic value。** 它回答一个更一般的 systems 问题：在线 resource manager 至少需要增加多少信息，才能真正改善 placement decision？

**Production value。** Runtime 团队可以用自己的 workload 判断某个 profiler、compiler analysis 或 scheduler integration 是否值得，而不是因为别人的 paper 有 speedup 就直接把整套机制搬进来。

**Failure condition。** 如果额外 evidence 相对 fault-only policy 几乎不能降低 regret 或 end-to-end cost，那么更简单的 UVM 行为应该继续作为默认方案。

## 哪些结果会改变这个判断？

现有证据只支持一个相对窄的结论：在 HBM oversubscription 下，GPU memory placement 会从 demand fault 之外的信息中获益，而近期系统分别从几个互不兼容的层次获得这些信息。它并不能证明应该用一个 universal placement policy 或一种 universal signal 替代 vendor UVM heuristic。

有三类结果会明显削弱本文的方向。第一，如果用当前 driver 和 hardware 重新做大规模 reproduction 后发现默认 UVM 已经接近各种 specialized policy，增加 contract 的价值会很小。第二，如果 sampled HBM observability、compiler semantics 与 scheduler knowledge 实际只服务完全不重叠的 workload，统一 evidence contract 的复用价值也会很低。第三，如果暴露 decision metadata 与 control path 的成本比它减少的 migration 还高，这个抽象就不该进入生产路径。

所以最值得先做的实验不是再写一个独立 eviction heuristic，而是做 fixed-budget comparison：让同一个 placement controller 逐级获得更丰富的 evidence，然后直接测这些信息到底有多少次改变了真正影响性能与 QoS 的 decision。

## 参考资料

- NVIDIA：[CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)，CUDA 13.x 文档，访问于 2026-08-29。
- NVIDIA：[CUDA Runtime API: Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)，访问于 2026-08-29。
- NVIDIA：[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)，Unified Memory page-fault tracing，访问于 2026-08-29。
- Linux kernel documentation：[Heterogeneous Memory Management](https://docs.kernel.org/mm/hmm.html)，访问于 2026-08-29。
- AMD：[HIP Unified Memory Management](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html)，访问于 2026-08-29。
- Pratheek B, Khushit Shah, Arkaprava Basu：[Observability-aided GPU Memory Oversubscription](https://www.csa.iisc.ac.in/~arkapravab/papers/ISCA26_ObservUVM.pdf)，ISCA 2026。
- Hyunkyun Shin, Seongtae Bang, Hyungwon Park, Daehoon Kim：[ARIADNE: Adaptive UVM Management for Efficient GPU Memory Oversubscription](https://doi.org/10.1109/HPCA68181.2026.11408564)，HPCA 2026；[Artifact](https://zenodo.org/records/17830000)。
- Yueqi Wang et al.：[OASIS: Object-Aware Page Management for Multi-GPU Systems](https://yueqiwang42.github.io/assets/pdf/papers/OASIS_HPCA25.pdf)，HPCA 2025。
- Pratheek B, Guilherme Cox, Jan Vesely, Arkaprava Basu：[SUV: Static Analysis Guided Unified Virtual Memory](https://www.csa.iisc.ac.in/~arkapravab/papers/MICRO24_SUV.pdf)，MICRO 2024。
- Weihang Shen, Yinqiu Chen, Rong Chen, Haibo Chen：[MSched: GPU Multitasking via Proactive Memory Scheduling](https://arxiv.org/abs/2512.24637)，arXiv:2512.24637，2026。
