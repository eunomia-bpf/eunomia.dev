---
date: 2026-08-31
title: "GPU 利用率能告诉运行时另一个内核还能不能放得下吗？"
description: "GPU 利用率描述最近有多忙，却不能直接回答新内核能否安全共存。本文提出面向具体候选任务的 allocatability contract。"
tags:
  - Daily Report
  - GPU
  - Runtime
  - Scheduling
  - Observability
research_question: "GPU 运行时要判断一个具体的新内核或任务能否与现有工作安全共存，需要哪些证据？"
source_cutoff: 2026-08-31
status: daily-report
---

# GPU 利用率能告诉运行时另一个内核还能不能放得下吗？

一块 GPU 当前显示 45% utilization，于是调度器把另一个对延迟敏感的 kernel 放了上去。这个数字看起来很宽松，像是还有一半以上的设备没有被使用。但新 kernel 可能仍然启动得很晚，也可能把原来的任务拖慢很多；如果两边还依赖同步或 collective，甚至可能出现无法继续推进的情况。

原因在于，这其实是两个不同的问题。"刚才这块 GPU 有多忙" 是对过去一段时间的测量；"这个具体任务现在能不能放进去" 是一次面向未来的 admission decision。后者取决于候选 kernel 的 register、shared memory、block shape、同步方式、显存需求以及调度约束，而这些信息并不包含在一个 utilization 百分比里。

随着 MPS、CUDA Green Contexts、MIG、推理服务多路复用、通信计算重叠以及集群级 GPU 共享逐渐成为常见运行方式，GPU 已经不只是一次跑一个 job 的加速器。运行时越来越需要回答一个更严格的问题：加入这份新工作以后，它是否真的有资源前进，同时又不会把现有工作推过不可接受的性能边界？

<!-- more -->

本文主张，共享 GPU runtime 应该暴露一个**面向具体候选任务的 allocatability contract**，而不是把一个 utilization 数字直接解释成剩余容量。这里的目标不是精确预测每个 kernel 的运行时间，而是给出一个更窄、也更可验证的结论：针对这个候选任务，目前哪些共存条件有明确资源保证，哪些只是在若干假设下可行，哪些根本无法保证。

这个问题与之前的 [GPU 显存放置证据](https://eunomia.dev/zh/research/gpu-memory-placement-evidence/) 不同。那篇报告研究 oversubscription 下什么时候应该迁移或淘汰页面；也不同于 [GPU 插桩安全契约](https://eunomia.dev/zh/research/gpu-instrumentation-safety-contract/)，后者关注 observer 会不会改变被测 kernel。这里的决策发生在执行之前：另一个 workload 到底应不应该被放进同一组物理 GPU 资源。

## Utilization 反映活动，不等于资源预留

NVIDIA 当前的 [Fleet Intelligence 文档](https://docs.nvidia.com/fleet-intel/data-collection-rationale/) 把 `DCGM_FI_PROF_SM_ACTIVE` 定义为采样窗口内，SM 至少有一个 warp 被分配的 cycle 比例；`DCGM_FI_PROF_SM_OCCUPANCY` 表示 resident warp 相对于理论上限的比例；`DCGM_FI_PROF_DRAM_ACTIVE` 则描述 device memory interface 有多长时间处于活动状态。这些指标很适合回答性能分析问题，因为它们能够告诉我们最近是哪类资源在忙。

但它们本身不会为下一个 kernel 预留任何资源。

时间平均会把这个差异放大。20% 的 SM activity，既可能来自一个 kernel 在采样周期的五分之一时间里几乎铺满所有 SM，也可能来自一个持续运行、但只占据较少 SM 的任务。两种执行历史可能得到相近的平均值，但第二个 kernel 在某一时刻能否马上开始执行，答案可能完全不同。

Occupancy 提供了更多信息，但 admission 仍然必须看候选 kernel。NVIDIA 当前 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 明确说明，resident block 和 warp 的数量受有限 register、shared memory 以及 block/warp 上限共同约束。Register allocation 还存在粒度效应，同样的 per-thread register 使用量，在不同 block size 下也可能得到不同 occupancy。[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf) 同样暴露了 per-SM register、shared memory、resident thread 和 block 等限制，而 occupancy 计算必须同时知道目标 kernel 自己需要多少资源。

因此，"这块 GPU 平均 occupancy 是 35%" 只是设备测量结果；"kernel B 可以在 kernel A 旁边每个 SM 再放两个 block，而且不会耗尽 register 或 shared memory" 才是针对候选任务的可行性判断。

## CUDA 的资源分区已经直接暴露了这个区别

当前 [CUDA Green Contexts 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html) 给出了一个很直接的生产场景。Kernel A 可以先占满所有可用 SM，导致后来的低延迟 kernel B 必须等待。Green Context 可以给 B 预留一组确定的 SM，使得 A 即使还有更多并行工作，也不能消费这部分资源。

同一份文档还对比了 MPS。默认情况下，多个 MPS client 会竞争可用 SM。`active thread percentage` 可以限制一个 client 最多使用多少比例的 SM，但这些 SM 具体是哪一些可以随时间改变。Green Context 则把 context 限定在创建时分配的具体 SM 上。CUDA 13.1 起，MPS 还支持显式开启 static partitioning。

这比 "utilization 低于某个阈值就继续放任务" 强得多，因为运行时把推测出来的容量变成了真实资源边界。

当然，这个保证也有范围。Green Context 文档说，预留 SM 可以让低延迟工作不必等待另一个 kernel 释放 SM，前提是没有其他资源约束。因此调度器仍然不能只看一个数字。Register、shared memory、thread-block cluster、HBM bandwidth、copy engine 和同步关系都可能决定名义上的 SM headroom 到底能不能被新工作使用。

## 对需要共同前进的 workload，错误 admission 不只是性能问题

两个彼此独立的 kernel 共存失败，通常表现为 latency 或 throughput 变差。但对依赖同步的 GPU 程序，错误资源判断可能直接破坏 forward progress。

当前 [NVSHMEM 文档](https://docs.nvidia.com/nvshmem/api/latest/using.html) 说明，当多个 processing element 通过 MPS 共享一块 GPU 时，如果想完整支持 synchronization 和 collective API，同一 GPU 上各 PE 的 active-thread percentage 总和需要不超过 100%。当总和超过 100% 时，CUDA 无法保证所有 PE 可以同时在 GPU 上运行，因此 point-to-point synchronization 或 collective 可能发生 deadlock。

这给出了一个很强的反例。历史 utilization 看起来很低，并不意味着运行时可以随便再加入一个 collective participant。某些 workload 在 admission 时就需要显式证明所有参与者能够同时前进。

2026 年 6 月的预印本 [Resource-aware Computation-Communication Overlap for multi-GPU ML Workloads](https://arxiv.org/abs/2606.09200) 从性能角度触及了同一个边界。作者通过提高计算 kernel 的 per-block shared-memory 使用量来主动限制其 residency，为通信 kernel 留出足够的 on-chip resource，再配合更高优先级的 communication stream 保证通信持续推进。在 A40、A100、H100 和 MI250X 上，论文报告总执行时间最高下降 25.5%。这里真正有效的机制并不是 "让 GPU 更忙"，而是给另一类工作留下它真正需要的资源。

更早的生产系统已经证明共享 headroom 很有价值。[AntMan](https://www.usenix.org/conference/osdi20/presentation/xiao) 在阿里巴巴的多租户 GPU 集群里共同放置深度学习 job，并动态缩放 memory 和 compute resource。论文报告 GPU memory utilization 提高 42%，compute-unit utilization 提高 34%，同时保持其公平性目标。今天更值得研究的问题不是要不要做 co-location，而是随着 GPU 暴露越来越多 partitioning 和 sharing 机制，运行时应该怎样表达一次 admission decision 的证据。

## 真正缺少的是面向候选任务的 headroom

实际系统里最容易犯的错误，是把 GPU 空闲容量建模成一个 device-level scalar。更合适的做法是先拿到候选 workload，再问当前资源能不能满足它的执行要求。

对于一次 kernel launch，候选任务可以携带这些信息：

- 每个 thread 的 register 数量，以及目标架构真正使用的 allocation granularity；
- 每个 block 的 static 和 dynamic shared memory；
- thread、warp、block 数量，以及 cluster 或 cooperative launch 要求；
- 预期 HBM footprint，以及是否要求一部分 memory 始终 resident；
- 当 DRAM、copy engine 或 interconnect 压力会决定结果时，对这些资源的需求；
- stream priority、Green Context、MIG 或其他 partitioning constraint；
- forward progress 是否依赖 peer kernel 或其他 processing element 同时 resident。

设备侧也应该是结构化状态，包括具体 SM partition、当前 resident-resource headroom、memory reservation、Green Context/MIG ownership、MPS limit，以及采样 telemetry 的新鲜度。

最终结果不需要伪装成一个非常精确的概率。一个实用 API 可以只返回少量明确状态：

```text
GUARANTEED_FIT       已预留资源足以满足候选任务的 contract
CONDITIONAL_FIT      在声明的 bandwidth / interference 假设下可行
BEST_EFFORT          telemetry 显示有 headroom，但没有 forward-progress 保证
REQUIRES_REPARTITION 需要 preemption 或重新划分资源后才可行
NO_FIT               某个硬资源或 progress constraint 已违反
UNKNOWN              证据过期，或者 backend 无法建立这个性质
```

这样，调度器在事故之后就能解释自己的决定。如果一个延迟敏感 kernel 只凭时间平均 utilization 被标成 `BEST_EFFORT` 后出现抖动，operator 会知道当时本来就没有强保证；如果运行时基于明确 SM partition 给出 `GUARANTEED_FIT`，结果仍然 miss SLO，就应该继续查 bandwidth、同步、launch latency 或 contract 中其他假设，而不是重新猜 utilization 阈值。

## 现有研究还缺什么

### Telemetry 与 admission 通常是两套接口

GPU monitoring 会提供 activity、occupancy、memory、clock 和 throttling；runtime API 会提供 kernel resource usage 与 partitioning control。很多 scheduler 最终只能用一个本地 heuristic 把两者拼起来，例如 "utilization 低于 60% 就继续放任务"。

缺少的是一条机器可读的 admission record，把这个具体候选任务的需求、当时的 device resource state 和最终决策放在一起。没有它，一次 bad placement 发生以后，很难区分究竟是 telemetry 过期、资源模型错了、遗漏了 interference，还是 scheduler policy 本身选错了。

一个有区分力的实验，是构造平均 utilization 完全相同但 spatial/temporal resource layout 不同的 replay。真正有价值的 contract 应该能区分 scalar threshold 看起来一样的 case。

### 时间平均会掩盖空间碎片和 phase 变化

SM activity 和 occupancy 之所以便宜，是因为它们把庞大的设备状态压缩成少量数字。但 co-residency 恰好可能需要被压掉的信息。短时间几乎占满全部 SM 的 burst，与长期只占小部分 SM 的 narrow kernel，可以产生相近平均值，但给第二个 workload 带来的启动机会和 tail latency 完全不同。

这里不一定需要 per-cycle tracing。runtime 可以只保留粗粒度 per-partition 或 per-epoch headroom，以及证据 age 和 variance。真正需要测的是，从一个 aggregate percentage 增加到一个小型 headroom distribution 后，admission accuracy 能提高多少。

### 硬资源可行性与性能干扰经常被混进同一个预测分数

Register、shared memory 和 partition ownership 可以形成相对硬的 feasibility boundary；DRAM contention、cache interference、power limit 与 scheduler interaction 更像连续的 slowdown risk。把两者都塞进一个 opaque score，会让错误很难解释。

更清晰的接口应把 hard constraint 与 performance-risk estimate 分开。runtime 可以证明候选 kernel 有足够 resident resource，同时声明其 bandwidth interference 风险仍然很高。如果系统使用 learned performance model，这个模型应该进入 `CONDITIONAL_FIT` 的假设部分，而不是覆盖可验证的 resource check。

### 依赖同步的 workload 需要比独立 kernel 更强的保证

NVSHMEM 对 MPS 的要求说明，simultaneous residency 有时是 correctness 和 forward progress 条件，而不只是 performance optimization。通用 GPU scheduler 目前很少把这种要求作为 candidate manifest 的一等属性。

评测应该加入 collective、producer/consumer kernel、persistent kernel 等可能等待 peer 的 workload。当 runtime 无法保证相关参与者同时前进时，正确行为应是拒绝 admission 或先 repartition，而不是依赖过去的 utilization 继续尝试。

## 兼具学术价值与生产价值的方向

### 1. 面向候选任务的 allocatability certificate

**缺口。** 现有 utilization telemetry 描述设备状态，而 admission 需要同时知道设备和具体任务。

**机制。** 增加一个 runtime query，输入 candidate resource manifest，输出 allocatability certificate。Manifest 记录 register、shared memory、block/cluster shape、memory residency、partition constraint 和 progress dependency；backend 将其与当前资源 ownership 和带时间戳的 headroom snapshot 结合。

Certificate 需要记录真正使用的 partition、检查过的 hard constraint、仍然依赖的假设、证据 age，以及前面列出的明确 outcome。CUDA backend 可以组合 compiler/runtime resource metadata、Green Context 或 MPS 状态、occupancy calculation 和少量 DCGM/CUPTI observation。其他 GPU backend 则用自己的 native resource descriptor 实现相同公开语义。

**与现有工作的差异。** CUDA occupancy API 可以估算一个 kernel 怎样映射到 SM，Green Context 能提供显式 SM partition；这里增加的是跨 workload 的 admission boundary，并且把决策证据保留下来。

**可实现产物。** 一个小型 runtime library 与 scheduler plugin，暴露 `can_admit(candidate, device_state)`；再定义 JSON certificate 格式和 post-incident replay 工具。

**评测。** 对比 utilization threshold、occupancy threshold、candidate-conditioned certificate 和受控实验得到的 oracle。工作负载系统扫描 register pressure、shared memory、block shape、Green Context/MPS partition 与 memory pressure。核心指标是 unsafe admission、unnecessary rejection、start-latency error、throughput loss 和生成 certificate 的开销。

**学术价值。** 它研究 heterogeneous-resource admission 有多少部分可以表达成可移植、可证伪的 contract，而不是 scheduler 私有分数。

**生产价值。** 集群与 inference scheduler 可以更积极地做 co-location，同时保留 operator 在 latency/throughput 出问题时能够检查的理由。

**失败条件。** 如果简单的 occupancy 加 utilization threshold 在同一批 workload 上拥有相同的安全 admission precision 和 SLO 结果，这套 certificate 没有足够价值。

### 2. Temporal / spatial headroom ledger

**缺口。** 一个平均值丢掉了 free capacity 位于哪里，以及它是否持续足够长时间。

**机制。** runtime 按短 epoch 维护一个 compact headroom ledger。它不保存所有硬件事件，只记录有限摘要，例如可用 SM partition、resident-resource headroom 的 minimum/percentile、DRAM/copy-engine pressure range 和每项 evidence age。当 resource shape 明显变化时切换 epoch。

这不是新的 profiler trace。它专门回答 admission："过去 10 ms 内，这组资源至少持续存在这么多 headroom，当前 uncertainty 是多少。" Candidate certificate 再检查自己的 required horizon 是否能与合适 epoch 匹配。

**与现有工作的差异。** DCGM 等工具主要为了 performance analysis 汇总 activity；这里保留的 temporal/spatial structure 只服务于 prospective admission，不追求完整 trace。

**可实现产物。** 一个由 runtime launch metadata 和现有 device telemetry 驱动的 headroom summarizer，以及一组平均值相同但 allocatability 不同的 synthetic / production replay trace。

**评测。** 构造相同 average SM activity/occupancy、但 burstiness、SM partition、resource fragmentation 与 phase length 不同的 workload pair。随着 ledger 从详细 epoch 压缩到一个 aggregate metric，比较 admission precision 与 tail latency。一个重要 ablation 是保留同样字节预算但去掉 spatial identity。

**学术价值。** 研究 recent device state 的最小充分表示，究竟需要保留多少结构才能可靠支持未来调度。

**生产价值。** scheduler 可以减少过度保守的 idle capacity，而不需要一直收集 fine-grained trace。

**失败条件。** 如果在相同 telemetry budget 下，temporal/spatial summary 对 admission 没有明显优于 rolling average，就不应该增加这层 ledger。

### 3. 专门制造错误 admission 的 counterexample benchmark

**缺口。** GPU scheduler 常用 aggregate utilization 与总吞吐评测，却可能看不到 "headline utilization 一样、正确 admission 结果相反" 的 placement。

**机制。** 构造一组 pair，主动保持相近 monitoring summary，同时改变真正控制 co-residency 的 hidden condition：

| 对照 | 相近的表面指标 | 不同的隐藏条件 |
| --- | --- | --- |
| bursty vs persistent work | average SM active | 短时间铺满所有 SM vs 持续占少量 SM |
| low- vs high-register kernel | average utilization | resident-block headroom |
| shared-memory-light vs heavy | average occupancy | per-block shared-memory feasibility |
| independent vs collective peers | SM utilization | simultaneous-progress requirement |
| partitioned vs contended contexts | device utilization | 是否存在 guaranteed SM ownership |
| compute-light vs bandwidth-heavy | SM activity | DRAM / interconnect interference risk |

每个 test 只问一个明确问题：在给定 SLO 或 progress requirement 下，现在是否应该 admit candidate B？benchmark 用受控执行建立 oracle，再评价 scheduler 的 admission decision，而不是只奖励更高 utilization。

**与现有工作的差异。** AntMan 证明 co-location 的收益，近期 overlap 工作证明主动保留 on-chip resource 的收益；这个 benchmark 专门测更早一步的决策：当前 evidence 是否真的足以支持这个 candidate 的 co-residency。

**可实现产物。** CUDA-first benchmark，包含 resource-swept kernel、Green Context/MPS variant、NVSHMEM progress test、可 replay 的 telemetry summary，并预留 AMD 或其他 backend adapter。

**评测。** 主要指标是 unsafe-admission rate、false rejection、SLO violation、progress failure、throughput 与 evidence cost。最关键的实验保持 aggregate utilization 几乎不变，只修改隐藏资源约束。如果 admission policy 仍然无法区分这对 case，它依赖的 signal 就不对。

**学术价值。** 这会把 allocatability 从经验 heuristic 变成可测量的 systems property，并明确展示 average utilization 在什么条件下信息不足。

**生产价值。** runtime 和 cluster 团队可以在新 GPU 架构、driver、partitioning mode 或 workload mix 上回归测试自己的 scheduler heuristic。

**失败条件。** 如果这些 counterexample 只存在于人工 microbenchmark，在代表性生产 workload 中完全消失，那么 allocatability 更适合作为 offline tuning 问题，而不值得成为 runtime abstraction。

## 哪些结果会改变这个判断？

本文假设共享 GPU 会长期承载 resource footprint 和 progress requirement 差异很大的 workload。如果生产 trace 表明实际 candidate 高度同质化，低 utilization 几乎总能预测安全 co-residency，那么 candidate-aware contract 确实会显得过重。

如果未来 GPU 硬件或 driver 提供了一个权威 admission primitive，本身已经同时考虑 register、shared memory、partition ownership、bandwidth 与 progress dependency，那么 runtime 最合理的做法应该是直接暴露这个 primitive，而不是在软件中重新推导 allocatability。

最强的检验仍然是同预算对比。固定 telemetry 成本，用简单 utilization threshold 和 candidate-aware admission 同时跑故意构造的困难 workload 与代表性生产 workload。如果两者在 unsafe admission、tail latency、forward progress 和 throughput 上没有实质差异，就保留简单方案；如果结果稳定分开，scheduler 就不应该继续把 utilization 当成容量，而应记录真正支持 placement decision 的证据。