---
date: 2026-08-31
title: "GPU 利用率能告诉你还能不能安全塞进一个任务吗？"
description: "GPU 利用率描述过去一段时间有多忙，却不能直接回答还能不能安全并行一个任务。本报告提出“可分配性”契约，把硬资源可容纳性、干扰预算和证据不确定性分开。"
tags:
  - Daily Report
  - GPU
  - 调度
  - 资源分配
  - 运行时
research_question: "什么时候 GPU 利用率足以支持新任务准入？如果不足，安全判断共置还需要哪些资源与干扰证据？"
source_cutoff: 2026-08-31
status: daily-report
---

# GPU 利用率能告诉你还能不能安全塞进一个任务吗？

GPU 监控面板显示一张卡的利用率只有 35%。队列里正好还有一个推理服务在等。最自然的调度决策是：既然这张卡还很“空”，就把第二个服务放上去，不要再开一张 GPU。

这个判断可能朝两个方向出错。正在运行的任务虽然平均利用率不高，却可能已经占住了新任务需要的寄存器、共享内存、workqueue、HBM 容量，或者在短时间内把显存带宽打满。反过来，一张看起来利用率很高的卡，也可能仍然有独立的空间分区、互补的资源需求，或者稳定的空闲阶段，可以安全容纳另一份工作。

真正缺少的不是一个“更准确的利用率”，而是**可分配性（allocatability）**：对于一个具体的新任务，在当前这张 GPU 上、当前这个时刻、给定明确的正确性和性能预算，它到底能不能被准入。

<!-- more -->

本报告的核心判断是：利用率应该继续作为观测信号，而不应该被当成准入证明。一个可用的 GPU 运行时至少要把三个问题拆开：

1. **新任务在物理上能不能放进去？** 寄存器、共享内存/LDS、可驻留 block 或 wave、显存、SM 分区、workqueue 等有限资源构成硬约束。
2. **放进去以后会不会互相拖慢到不可接受？** DRAM、cache、互联、执行管线、功耗以及时间重叠形成软干扰，需要同时看两个任务。
3. **这个判断的证据有多可靠？** 时间窗口平均值、过期 profile 和缺失计数器都应该降低置信度，而不是被自动翻译成“还有这么多空闲容量”。

这个区别对集群调度、模型服务、GPU serverless 以及本机多进程共享都适用。它也和前两篇 GPU Daily Report 的边界不同：[GPU 内存放置](https://eunomia.dev/zh/research/gpu-memory-placement-evidence/)问数据该放在哪里，[GPU 动态插桩安全](https://eunomia.dev/zh/research/gpu-instrumentation-safety-contract/)问观察者会不会改变被观察程序。这里问的是另一件事：观察到的忙闲程度，是否足以证明“还能再放一个任务”。

## 一个利用率百分比只是活动平均值，不是资源库存表

NVIDIA 当前的 [DCGM Profiling 文档](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html) 对这些计数器的语义写得很清楚。`PROF_SM_ACTIVE` 表示一个采样窗口里，SM 上至少有一个 warp 活跃的时间比例，再对所有 SM 求平均。同样的 20%，既可能来自五分之一的 SM 在整个窗口里一直工作，也可能来自全部 SM 只工作了五分之一时间。`PROF_SM_OCCUPANCY` 则单独表示驻留 warp 相对硬件上限的比例，`PROF_DRAM_ACTIVE` 描述显存接口活动。DCGM 还明确提醒，只看某一个高计数器，不能证明程序一定是 compute-bound 或 memory-bound。

这正是为什么一个 headline 利用率不适合直接做准入判断。调度器想知道的是“另一个具体任务能否和它一起运行”，但这个数字只告诉我们，原有任务在经过时间和空间平均之后做了多少活动。

设想两张卡在过去一秒都显示 40% SM activity：

- A 在 400 ms 内几乎占满所有 SM，剩下 600 ms 空闲；
- B 用一个窄的 persistent kernel 长时间占住约 40% 的 SM。

平均值完全一样，但第二个任务看到的机会不一样。一个要求马上启动的低延迟 kernel 可能更喜欢 B 的空间余量；一个可以等稳定空窗的 batch 任务反而可能适合 A。单个百分比无法恢复它是怎样在时间和空间上形成的。

Occupancy 信息更多，但仍然不是准入答案。DCGM 把 occupancy 定义为驻留 warp 相对最大值的比例，同时指出 occupancy 越高并不总是越好。一个显存带宽受限的 kernel 和一个计算受限的 kernel 可以有类似 occupancy，却给共置任务留下完全不同的干扰面。

因此生产上应该把一句话记清楚：**利用率描述的是已经发生的工作，可分配性讨论的是还没被放进来的工作。** 从前者推到后者，必须知道新任务需要什么，以及它会和现有任务共享什么。

## “能不能放进去”是离散约束，不是把 60% 空闲直接相加

GPU 执行资源不是一池连续容量，不能把“60% 没用”直接理解成还有 60% 可以分。NVIDIA 当前的 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) 说明了寄存器数、每 block 线程数以及分配粒度如何共同决定一个 SM 能驻留多少 block。同样的每线程寄存器数，只因为 block 大小不同，就可能跨过不同的资源分配边界并得到不同 occupancy。

AMD 的术语不同，但现象一样。当前 [ROCm workload optimization 文档](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html) 根据 VGPR、LDS 和每 workgroup 的 wave 数计算 occupancy。MI300X 的例子还展示了寄存器会按硬件粒度向上取整，几个寄存器的变化就可能让可驻留 wave 数直接下降。更新的 [ROCm Compute Profiler 文档](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/rdna/wgp.html) 也把 VGPR 和 LDS 作为限制 wave/workgroup 驻留的资源来报告，而不是把它们合并成利用率。

这些属于硬可行性。如果当前资源状态让新 block 无法获得足够的寄存器或共享内存，那么上一秒 SM 有多闲并不重要。同样，某个瞬间还有很多 HBM 空闲，也不能证明第二个模型一定安全，因为第一个任务可能有弹性的 working set，或者无法在新服务的延迟 deadline 内完成内存回收。

所以“剩余容量”更合理的表示是一个向量，而不是一个标量。实际运行时至少可能需要考虑：

```text
SM / CU 分区与放置
可驻留 block / wave 容量
寄存器分配档位
每 block 的 shared memory / LDS
workqueue / connection 资源
HBM 预留与可回收性
DRAM / cache 干扰敏感度
PCIe / NVLink 需求
功耗与热余量
```

不同后端不会都暴露这些维度，而且其中一部分是共享干扰信号，不是真正可以 reserve 的资源。但正确做法应该是把能力缺口和不确定性写出来，而不是因为 API 看不到，就把所有维度重新压成一个 utilization。

## 即使 SM 已经分区，也不代表并行执行有完整保证

CUDA 13.2 的 API 已经很直接地暴露了“拥有某个资源”和“真的可以稳定并行”之间的区别。当前 [CUDA Runtime execution-context 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html) 允许程序根据指定的设备资源创建 Green Context，包括彼此不重叠的 SM 分区以及 workqueue 配置。

但 NVIDIA 同时明确写道：即使两个 Green Context 的 SM 分区完全不重叠，也**不能保证**其中的 kernel 一定并发运行或有 forward-progress guarantee，因为还可能有其他资源产生依赖。使用平衡、尽量不重叠的 workqueue 可以提高避免干扰的机会，但要获得更强保证，还需要未来暴露更多资源类型。

这给运行时设计提供了一个很有价值的边界。即便调度器已经拥有比利用率更强的信号，比如一个明确的 SM 空间分区，也不能用单一资源维度代表整个共置契约。可分配性判断最好至少区分：

- **硬可容纳（hard fit）**：对于运行时知道并且能约束的资源，可以证明这些条件满足；
- **干扰风险**：共享、不可 reserve 或随时间变化的资源，仍然可能破坏性能预算；
- **未知（unknown）**：后端没有足够的观测或隔离能力，不能支持一个高置信度准入结论。

把 `unknown` 当成正式结果很重要。否则系统很容易把“API 没有告诉我”逐渐偷换成“这里没有约束”。

## 已有 GPU 共享系统实际上都在管理比利用率更多的东西

近年的生产系统和研究系统反复得到类似结论：真正共享 GPU 时，最终都需要加入 workload-specific 的资源控制，而不是只看一条活动曲线。

USENIX ATC 2025 的 [SIRIUS](https://www.usenix.org/conference/atc25/presentation/wang-jiali) 把推理和训练共置。它让延迟敏感的推理优先使用资源，训练再消费剩余部分。但 SIRIUS 并不是看到低利用率就直接塞训练任务，而是会动态改变训练内存消耗，显式执行显存回收和 handover，并按 SLO 做内存重新分配以避免 thrashing。

同一届 ATC 的 [KRYPTON](https://www.usenix.org/conference/atc25/presentation/zhang-shulai) 用 kernel-space command-buffer interception 做 GPU 共享，并采用时空结合的资源共享来提供性能保证，而不是只依赖固定分区。它再次说明，调度和隔离来自共享机制本身，不来自某个 utilization counter。

更早但规模很大的 [AntMan](https://www.usenix.org/conference/osdi20/presentation/xiao) 在阿里生产 GPU 集群中把调度器和训练框架一起设计，使训练任务可以动态缩放内存和计算资源，再把真正能够让出来的部分交给其他任务。所谓“spare GPU resource”，只有在运行时知道怎样让一个任务释放资源给另一个任务之后才有意义。

2026 年 7 月的预印本 [Roomie](https://arxiv.org/abs/2607.16784) 从模型服务角度补了另一个证据。作者认为 aggregate resource profile 会丢掉 kernel 时间重叠，因此用 per-kernel resource configuration 和 occupancy-based analytical model 估计共置干扰。论文报告在其测试中，相比基线减少了 SLO violation，同时保持相近或更好的 goodput。这个设计未必能直接泛化到所有 GPU 工作负载，但它说明了一个方向：准入判断需要知道“将要进来的 workload 长什么样”，而不只是现有 workload 的平均利用率。

这些工作并不意味着可以设计一个通用 allocator 替代所有 workload-aware 调度。它们支持的是一个更窄的结论：**利用率的信息损失太大，不适合作为共享正确性的边界。**

## 现有工作仍然薄弱的地方

### 硬可行性和软干扰经常被压成同一个分数

很多调度器会把 utilization、显存占用，也许再加 occupancy，组合成一个 “GPU load” 分数。用它排序设备很方便，但它把两种完全不同的失败混在了一起。

硬可行性失败意味着新任务根本拿不到所需资源或隔离边界。软干扰失败意味着它可以执行，但共置后延迟、吞吐或 slowdown 超过预算。这两种失败应该给出不同解释，也应该触发不同恢复动作。

一个有区分度的实验可以专门构造两类 workload pair：一类物理上完全能共置，但两个任务同时打满 DRAM；另一类近期 utilization 很低，但寄存器、共享内存或分区约束让新 kernel 无法获得有效驻留。单一 score 很难解释它们，而一个合理的运行时契约应该能把这两类失败分开。

### 调度器通常只知道“正在跑的任务”，却不知道“准备进来的任务”

大多数 telemetry 描述的是已经在 GPU 上运行的 workload，但准入问题恰好是关于尚未开始的 workload。即使把 resident workload 测得非常准确，如果没有新任务的资源 envelope，调度器仍然只解了一半问题。

模型服务可以提前 profile 已知 kernel，一般 GPU runtime 更麻烦。JIT kernel、动态 shape、CUDA Graph 变体、用户扩展都可能让资源需求变化。系统需要静态或 profile 得到的 envelope，同时要在 binary、shape family 或运行 phase 变化时明确把旧 envelope 标成过期。

### 平均计数器丢掉了决定干扰的时间重叠

一个窗口平均 40% 的 DRAM activity，可能是持续平滑的中等流量，也可能是几次短时间完全饱和。第二个 workload 遇到这两种情况，结果会很不一样。SM activity、NVLink 和功耗同样存在这个问题。

因此目标不应该只是把所有计数器采得更快。更高频的平均值仍然可能没有 overlap 的因果结构，却会增加监控成本。系统应该保留真正和准入问题相关的 phase 或 burst 证据。

### 准入失败之后的结果没有总是回流到容量模型

系统看到 SLO violation 后可以迁移或 throttle workload，但这次失败本身也是高价值证据。如果运行时在某个资源状态下把 B 放到了 A 旁边，并马上看到 3 倍延迟，这个结果应该更新以后相同 kernel、shape 和 resource regime 的可分配性估计。

如果没有这个反馈环，调度器会反复犯同一个“低利用率就是有空位”的错误。

## 同时有学术与生产价值的方向

### 1. 用“可分配性证书”替代利用率阈值

**缺口。** 调度器有很多活动计数器，却没有一个机器可读对象解释：某个具体新任务为什么能放进去、哪个资源是限制因素、还有哪些信息未知。

**机制。** 为每个待准入 workload 维护版本化的资源 envelope，再和当前设备资源快照进行匹配。Envelope 可以组合静态 kernel 属性、离线 profile、显存 reservation、runtime phase 和声明的 SLO。结果形成一个 **allocatability certificate**：

```text
workload_version
hard_fit: yes | no | unknown
limiting_resource
resource_reservations
shared_contention_dimensions
evidence_timestamp
profile_coverage
predicted_slowdown_range
confidence
expiry_or_revalidation_trigger
```

这里故意把 `hard_fit` 和 slowdown prediction 分开。CUDA Green Context 的资源描述符或者 MIG 一类硬分区，如果存在，可以增强“硬约束”部分；kernel attribute 和 occupancy calculation 可以提前拒绝明显放不进去的组合。DCGM 或 ROCm counter 则用于估计共享干扰，不冒充资源 reservation。

**相对现有工作的增量。** 现有 GPU scheduler 常常在内部维护 placement score 或模型 profile。这里新增的不是另一个打分公式，而是把“为什么准入、缺什么证据、哪个资源限制了它”变成跨后端都可检查的对象。

**可交付原型。** 一个小型 runtime service，加 CUDA 和 ROCm adapter。类似 `gpu-fit <device> <workload-profile>` 的 CLI 返回 admit/reject/unknown 和具体限制条件，而不是只给一个 load score。

**评估。** 构造不同 workload pair，分别扫描 register pressure、shared memory/LDS、HBM footprint、DRAM intensity、tensor/FP pipeline 以及 burstiness。比较 utilization-only、utilization+memory、occupancy-based 和 certificate-based admission。指标包括 false admit、false reject、决策延迟、最终吞吐和 SLO violation。

**学术价值。** 可以直接研究 GPU 共置能否找到一条有用的证明边界，把离散资源可行性与统计干扰预测分开。

**生产价值。** 运维人员能看到为什么一张“看起来很闲”的 GPU 不能再接服务，调度器也能区分“换一张卡”和“先补证据”。

**失败条件。** 如果简单的 utilization+memory 阈值已经能跨架构、跨 workload 类型达到和完整证书一样的准入准确率，那么新增契约没有必要。

### 2. 两阶段准入，再加一个有预算的干扰探针

**缺口。** 硬资源检查只能证明两个 workload 物理上可能共存，不能证明它们共享 DRAM、cache、互联、执行管线或功耗预算后仍满足延迟和吞吐目标。

**机制。** 把 admission 分成两步。第一阶段根据硬约束先排除不可能的 placement。只有通过这一关的候选才进入第二阶段，用历史 kernel-pair 模型加一个很短、严格受限的 canary 来估计干扰。这个 canary 可以放在预留 slice、低优先级 stream，或者另一个受控执行窗口里。

Canary 不能退化成无限制 benchmark，而要带明确风险预算和回滚条件：

```text
max_probe_time
max_requests_exposed
max_slowdown_on_resident_workload
required_confidence
rollback_on_slo_violation
```

如果在预算内仍拿不到足够证据，结果保持 `unknown`，调度器回退到隔离运行或者换设备，而不是把“没测出来”当作安全。

**相对现有工作的增量。** Roomie 用 kernel configuration 预测 model-serving 干扰，SIRIUS 和 AntMan 会主动重塑共置 workload。这里关注的是更通用的准入边界：先证明 hard fit，再只为真正不确定的共享维度花一小笔在线测量预算。

**可交付原型。** 一个可插拔 interference estimator、一个保守 canary executor，以及按 kernel/workload version、shape family、device type 和 resource regime 索引的 pairwise evidence cache。

**评估。** 在 phase change 和未见过的 workload pair 下比较无 probe、固定离线 profile、纯 analytical prediction、纯 canary 和 hybrid admission。主要看 false-admit、probe 本身造成的 SLO 影响、适应时间，以及重复 pair 多快可以不再重新 probe。

**学术价值。** 把 online profiling 变成一个带明确风险上限的 admission-control 实验，而不是持续不断的监控流。

**生产价值。** 新 workload pair 不需要永远保守拒绝，也不需要因为 utilization 低就乐观放行，而是可以在可控风险下试一次。

**失败条件。** 如果短 canary 对 resident workload 的扰动已经大到失去预测意义，或者干扰变化快到历史证据无法复用，那么在线 probing 不是合适机制。

### 3. 专门攻击“这张 GPU 还有空位”判断的反例 benchmark

**缺口。** 很多 GPU scheduler 最后报告平均 throughput 或 cluster utilization，但这些指标不能告诉我们“准入信号本身是不是可信”。

**机制。** 构造 headline 指标相同、真实可分配性不同的成对场景：

| 面板上看起来相同 | 隐藏差异 | 真正要回答的问题 |
| --- | --- | --- |
| 40% SM activity | 窄 persistent workload vs 全 GPU 的周期 burst | 低延迟 kernel 能否马上启动？ |
| 50% occupancy | register-limited vs DRAM-saturating kernel | 新 kernel 会不会超过 slowdown budget？ |
| 40% HBM used | 可回收训练 buffer vs 不可快速回收 working set | 能否在 deadline 内拿到显存？ |
| SM 集合互不重叠 | workqueue 或其他共享依赖仍存在 | 两边能否持续并发前进？ |
| 相同 DRAM 平均值 | 平滑流量 vs 短时饱和 burst | overlap 时 p99 延迟能否保持？ |

Benchmark 先通过受控 co-run 得到 ground truth，再要求调度器在看不到答案的情况下做 admit/reject。核心指标不是最终平均 GPU utilization，而是 false admission、false rejection、p99 slowdown violation、time-to-admission，以及对 limiting resource 的解释是否正确。

**相对现有工作的增量。** 现有 colocation 系统证明某种共享机制可以提高利用率或 SLO compliance。这里测试的是“凭什么说 GPU 有 spare capacity”这一层证据，并主动制造平均 telemetry 无法区分的反例。

**可交付原型。** 一套 CUDA-first 测试，能做的部分再提供 ROCm 对应案例；包括 workload-profile manifest、可复现 phase schedule，以及 DCGM、ROCm Compute Profiler、MPS/Green Context 和上层 scheduler adapter。

**评估。** 最重要的 ablation 是逐层增加信息：只看 GPU utilization；再加显存；再加 occupancy；再加静态 kernel resource；再加 phase；最后加入 pairwise interference evidence。这样可以直接看到哪些证据真正减少错误判断，哪些 telemetry 只是看起来更丰富。

**学术价值。** 把“GPU 还有剩余容量”从 dashboard 解释变成可证伪的 systems claim。

**生产价值。** Scheduler 团队在换 GPU 架构、driver 或 sharing mode 时，可以先回归测试，再决定是否继续沿用旧机器上的准入阈值。

**失败条件。** 如果这些反例在真实 scheduler 中根本不会改变 admission outcome，或者很小的一组固定 counter 已经可以把它们完全分开，那么 benchmark 应该收缩到那组更简单的证据契约。

## 哪些结果会改变这个判断？

最强的反方观点是：运维系统根本不需要一个新的 runtime abstraction。也许把现有的 SM activity、occupancy、DRAM activity、free HBM 和静态 kernel resource attribute 组合起来，就已经足够准确地预测共置。Roomie 最近的结果在已知 model-serving workload 和 kernel profile 的范围内，确实给这个方向提供了一些证据：occupancy-based analytical model 可以有效预测干扰。

这个反方观点应该被直接实验验证。如果一组简单、厂商正式支持的 metric vector，可以跨 CUDA 和 ROCm、跨 phase、跨未见过的 kernel pair、跨不同 sharing mode，同时低 false-admit 地预测 hard fit 和 SLO compliance，那么显式 certificate 和 bounded canary 层大部分只是在增加 plumbing。

另一个会削弱本文结论的变化，是 GPU 平台未来提供更强的资源 reservation。当前 CUDA Green Context 文档已经明确承认：SM 分区互不重叠仍不能保证并发，因为还有其他资源共享。如果未来 API 能把真正重要的资源都变成可 reserve 对象，并给出强 forward-progress 或性能隔离保证，那么“可分配性”可以更多地变成直接资源查询，而不是统计推断。

在那之前，“35% utilized”当然仍然是有价值的运维信号，但它不是“下一个 workload 能不能放进来”的答案。一个调度器最好明确告诉我们：什么已经证明能放，什么可能互相干扰，以及什么它现在还不知道。
