---
date: 2026-08-18
title: "eBPF 能把内存开销归因到真正使用的页面吗？"
description: "内存分配量、RSS 和页面热度回答的是不同问题。本报告分析 eBPF 如何把 mmap、malloc 等分配来源，与真正被访问、缺页、回收、迁移和消耗带宽的页面关联起来。"
tags:
  - Daily Report
  - eBPF
  - 内存分析
  - Linux 内存管理
  - 可观测性
  - 性能分析
research_question: "eBPF 内存分析器如何把应用的分配来源，与真正被访问、缺页、回收、迁移或产生内存访问开销的虚拟页和物理页关联起来，同时避免假设每个页面始终只有一个稳定所有者？"
source_cutoff: 2026-08-18
status: daily-report
---

# eBPF 能把内存开销归因到真正使用的页面吗？

一个服务启动时预留了 64 GiB 虚拟地址空间。正常流量下，它真正访问的只有 3 GiB，常驻集大约 2 GiB；一部分冷页会被回收后再次缺页载入，一部分文件页与辅助进程共享，NUMA 平衡还会周期性迁移页面。堆分析器能告诉我们哪些调用栈申请了内存，`/proc/<pid>/smaps` 能告诉我们各个映射当前常驻多少，DAMON 能估计哪些地址区间更热，硬件采样还能找到代价较高的内存访问。

这些数字都对，但它们并不是同一个问题。

当机器出现高 RSS、回收压力或内存带宽瓶颈时，工程师通常真正想问的是：**现在机器正在为哪些应用分配、缓存对象或运行时资源付出物理内存和访问成本？** 一个调用栈申请的内存可能从未被触碰；真正触发缺页的线程可能并不是创建映射的线程；一个物理页可能发生迁移、拆分、共享或写时复制；一次硬件采样可以给出访问地址，却不一定知道这个地址最初属于哪个分配对象。

<!-- more -->

本报告的判断是，eBPF 内存分析还缺一层位于“分配追踪”和“页面活动”之间的来源关联。更合适的分析单位不是单纯的“已分配字节”或“常驻页面”，而是**从应用分配开始，经过带生命周期的虚拟区间，再连接到页面或 folio 的生命周期以及访问证据**。

Linux 已经分别提供了其中大部分信息。真正缺的是跨层关联方式，以及一套能区分精确归因、采样估计和无法确定结果的评测方法。

这是 **eBPF Observability and Profiling** 系列的第一篇。站内已有的 [eBPF memleak 教程](https://eunomia.dev/zh/tutorials/16-memleak/) 会追踪分配和释放调用栈；这里继续追问下一层：内存申请完成以后，哪些部分最终变成了真实工作集和机器成本？

## 分配量、常驻量和实际使用量是三种不同测量

分配追踪观察的是应用意图。Eunomia 的 `memleak` 示例会用 uprobe 追踪 `malloc`、`calloc`、`realloc`、`mmap` 以及对应的释放路径，记录地址、大小和调用栈，并按尚未释放的分配做汇总。这非常适合定位泄漏或长期未释放对象。

但是它无法仅凭分配记录判断这些字节是否真正进入了物理内存。

Linux 从另一层提供常驻信息。[`/proc` 文档](https://docs.kernel.org/filesystems/proc.html) 区分虚拟内存大小和 `VmRSS`，`smaps` 还能按映射给出更细的内存统计。文档同时说明，普通 RSS 统计为了扩展性会异步维护，而扫描 `smaps` 可以获得更精确、但成本更高的快照。因此，常驻集回答的是“现在有哪些映射占用了常驻内存”，它本身并不会告诉我们“哪个分配调用栈让这些页面变热”或者“哪个页面正在制造带宽压力”。

Linux 也有直接观察页面使用情况的机制。[Idle Page Tracking](https://docs.kernel.org/admin-guide/mm/idle_page_tracking.html) 可以标记页面为空闲，并在内核观察到引用后清除空闲状态，因此能估计工作集大小。[DAMON](https://docs.kernel.org/mm/damon/design.html) 会采样访问频率和访问模式持续时间，并把相邻页面合并成区域来限制开销。DAMON 明确在空间精度与可控开销之间做权衡；当一个区域中的页面并没有相似访问模式时，区域采样会降低结果质量。

再往下，`perf mem` 和 perf 的 memory mode 可以在硬件支持时采样数据地址、访问延迟、TLB 和缓存信息，有些平台还能记录物理数据地址。这更接近真实 load/store 的成本，但它仍然是采样，而且依赖具体硬件。

因此，内存分析实际上包含几层不同事实：

| 层次 | 最适合回答的问题 | 单独使用时缺少什么 |
| --- | --- | --- |
| allocator / `mmap` 追踪 | 谁申请了这个地址区间和大小？ | 这些字节是否常驻或真正被访问 |
| `/proc/<pid>/smaps` | 哪些映射当前常驻、如何计费？ | 分配调用栈和访问强度 |
| Idle Page Tracking | 一个时间窗口里哪些页面被引用过？ | 应该把成本归到哪个应用对象 |
| DAMON | 哪些地址区域经常或最近被访问？ | 所有模式下的精确逐页来源 |
| `perf mem` / PMU 采样 | 哪些采样访问更昂贵、发生在哪里？ | 完整覆盖以及稳定的分配身份 |
| 页面生命周期 tracepoint | 内核何时分配、回收或迁移页面？ | 最终由哪个应用对象触发这段生命周期 |

如果分析器把其中任意一列直接叫成“内存使用”，在一部分工作负载上就会产生误导。

## 物理页面身份不等于应用所有权

一个直觉做法是用 PFN 把所有数据连接起来。Linux 的一些诊断接口会暴露 PFN，`mm_page_alloc` tracepoint 也记录页面分配时的 PFN。内核的 [page_owner](https://docs.kernel.org/mm/page_owner.html) 还能保存物理页分配时的调用栈、order 等信息。

问题在于，page_owner 描述的是**谁在内核中分配了这个 page**，它并不天然等价于“哪个用户态 `malloc` 应该承担成本”。文件页可以被多个进程共享；`fork` 后的匿名页在写时复制发生前也是共享的；透明大页把多个 base page 组合成 folio，之后还可能拆分；回收会让原来的物理页离开常驻集，后续缺页可能为同一个虚拟地址建立新的物理 backing；NUMA 迁移则会移动数据，而应用看到的分配仍然没变。

现有 tracepoint 能看到这些变化的一部分，但它们不是一个统一的页面 lineage API。[`mm_page_alloc`](https://github.com/torvalds/linux/blob/master/include/trace/events/kmem.h) 提供 PFN、order、GFP flags 和 migratetype；[`mm_vmscan_write_folio`](https://github.com/torvalds/linux/blob/master/include/trace/events/vmscan.h) 在回收写回时能提供 folio PFN，其他一些回收事件则是聚合统计；[`mm_migrate_pages`](https://github.com/torvalds/linux/blob/master/include/trace/events/migrate.h) 提供一批迁移的成功/失败数量、模式和原因，而不是每个页面从旧 PFN 到新 PFN 的完整对应关系。

因此，一个实用归因模型不应该把 PFN 当作根身份。应用侧更稳定的身份更接近：

```text
(进程或地址空间身份,
 分配或映射身份,
 generation,
 虚拟地址区间)
```

物理 backing 只是挂在这个身份上的、有明确生命周期的边。

这直接关系到正确性。如果分析器因为某个物理地址曾经属于调用栈 A，就一直把后续采样成本算给 A，那么在 unmap、地址复用、写时复制或迁移后都可能产生静默误归因。generation 不是实现细节，它是避免旧身份穿过地址复用的必要边界。

## 关联模型应该保留歧义，而不是虚构唯一所有者

真实内存经常存在多个合理消费者。共享库、共享内存、page cache、写时复制都会打破“一个页面只有一个 owner”的假设。

更合适的模型是一张小型 provenance graph：

```text
分配 / 逻辑资源
        |
        v
虚拟区间 generation
        |
        +---- 精确映射/缺页边 ----> page 或 folio generation
        |                            |
        |                            +---- 回收 / 迁移生命周期
        |
        +---- 采样访问边 ----------> 访问权重 / 延迟 / 带宽
```

每条边都要带上证据类型。allocator uprobe 可以在某个时刻精确建立地址区间；页面缺页 hook 可以证明某个虚拟地址获得了 backing；DAMON 可以为区域提供采样访问权重；PMU 内存采样可以提供部分 load/store 证据；共享映射则允许一个 page generation 同时连向多个虚拟区间，而不是强行选一个 owner。

最终报告可以把下面几类成本分开：

- **预留或申请字节**：来自分配、映射意图；
- **常驻字节**：来自当前映射和页表状态；
- **被访问的工作集**：来自一个时间窗口内的访问证据；
- **回收和迁移活动**：来自页面生命周期；
- **采样内存成本**：例如访问延迟、NUMA 位置或访问权重；
- **无法归因或多重归因的成本**：证据不足时保持显式，而不是硬分给一个对象。

这比把所有指标压成一个“memory usage”数字更适合故障分析。

## 现有研究还缺什么

### 分配分析通常停在 VM 生命周期之前

堆分析器和 eBPF allocation profiler 很擅长解释尚未释放的分配。它们天然使用“地址 + 大小 + 调用栈”作为 key。页面一旦经历缺页、回收、迁移、重新映射或共享，仅靠原来的分配记录就不足以解释机器正在承担的成本。

缺少的能力，是维护“分配区间”和 VM/page 生命周期之间的关系。这个关系必须能处理地址复用，也必须能表达共享 backing，而不是强行指定一个 owner。

最直接的实验是 reserve-versus-touch：预留几十 GiB，只访问一个可控子集，再让其中一部分被回收，最后把同一个虚拟区间释放并重新用于另一类对象。正确的分析器应该始终把预留、常驻、已访问和地址复用后的新分配区分开。

### 页面热度工具缺少应用分配来源

Idle Page Tracking 和 DAMON 已经能很好地回答工作集问题。DAMON 特别适合在开销和精度之间做可配置权衡，也能按区域观察访问频率和 age。它们的目标本来就是描述内存行为，而不是记录每个字节来自哪个用户态调用栈或运行时对象。

生产排障中，后面的来源关联往往才决定下一步动作。“这个 2 GiB 区域很热”不如“这个热区来自 cache shard 创建路径 X，并且大部分已访问页面属于十分钟没有服务请求的 shard”直接。

关键实验是比较两种方案能否做出不同的优化决策：只用 DAMON 区域统计，和加入 allocator/runtime provenance。如果后者没有带来更好的决策，就没有必要增加这层追踪。

### 回收和迁移事件还不是完整逐页 lineage 接口

Linux tracepoint 暴露了许多有价值的 VM 事件，但它们并不是为了组成一个统一页面 lineage 协议而设计。有的事件带 PFN，有的只给聚合数量；为了补齐信息，eBPF 还可能需要挂到 BTF 可见的内部函数，而这些函数的稳定性与 tracepoint 不同。

因此生产实现必须给 hook 标注稳定性，并在无法获得精确 lineage 时明确降级。否则同一个分析器在两个内核版本上可能使用不同证据，却输出看起来一样精确的结果。

可以用跨版本回放来验证这一点：在多组受支持内核上运行同一套内存 workload，要求分析器要么给出可比较的归因，要么明确报告 evidence loss，不能自动补成“精确结果”。

### 内存带宽归因天然受采样和硬件限制

硬件内存采样的价值在于它看到真实访问，而不是分配意图。但它从定义上就是不完整观测。支持哪些 load/store 事件、能否得到物理地址、采样 skid 和 memory-source 字段，都依赖架构。

因此带宽归因必须同时给出覆盖和置信信息。只观察到采样 load 时，就不能把采样权重包装成精确“带宽占比”。

一个可区分方案好坏的实验，是使用地址区间和访问次数都已知的内存生成器，比较 sequential、random、NUMA local 和 remote 模式下的采样归因误差。

## 兼具学术价值与生产价值的方向

### 1. 带生命周期的 allocation-to-page provenance ledger

**缺口。** 现有工具分别观察分配、常驻状态和 VM 事件，却缺少稳定的跨层关系。

**机制。** 用 eBPF collector 加用户态 join engine。uprobes 或运行时探针记录 allocator 对象及其虚拟区间；系统调用和 VM hook 记录 `mmap`、`brk`、`munmap`、缺页以及必要的映射变化。每个虚拟区间都有 generation，地址被释放再复用时必须创建新身份。如果能精确观察物理 backing，再建立有生命周期的 page/folio generation 边。共享和写时复制要产生多个或后继边，不能简单覆盖 owner。

数据模型还要区分 exact、inferred 和 sampled evidence，并在 unmap、迁移、回收和进程退出后及时淘汰旧 page identity。

**与已有工作的差别。** 它不是替代 DAMON 或 page_owner，也不只是把 memleak 做得更复杂。DAMON 继续负责访问行为，page_owner 继续解释内核页面分配；新机制负责把这些事实重新关联到应用分配来源。

**可实现产物。** 一个开放 schema、一组使用稳定 tracepoint 和少量 BTF hook 的 CO-RE BPF collector、glibc 和一个托管运行时的 allocator adapter，以及一个能输出 reserved、resident、touched、reclaimed、migrated 等 profile 的用户态重建库。

**评测。** 使用 reserve-versus-touch、`malloc` arena、文件映射、`fork` + COW、THP、reclaim、swap 和 NUMA migration 等确定性 workload，与 allocation-only eBPF、`smaps`、page_owner 和 DAMON 对比。核心指标不是图是否好看，而是对 ground truth 的 attribution precision/recall、丢事件率、CPU 开销、BPF map 内存和重建成本。

**学术价值。** 它研究的是一个可泛化问题：虚拟身份比物理 backing 活得更久时，资源来源怎样保持正确。

**生产价值。** SRE 可以从“这个进程 RSS 很高”继续定位到真正形成工作集的 allocator stack、缓存或运行时资源。

**失败条件。** 如果现实 workload 中，DAMON region 加普通 allocation stack 已经能以更低成本做出同样诊断，那么逐页 lineage 不值得长期运行。

### 2. 带明确置信度的 access-weighted attribution

**缺口。** RSS 会给一个冷页和一个产生数百万次访问的热页相同权重。

**机制。** 把 provenance ledger 与 DAMON 和硬件支持下的 `perf mem` 采样关联。DAMON 提供区域级访问频率和 age；PMU 样本在可用时提供地址、延迟、memory source 或物理地址。join engine 把这些权重归到 allocation/resource identity，同时保留 sampling rate、缺失字段、lost sample 和歧义。

输出应该是估计分布，而不是假精确。例如，“调用栈 X 对采样到的 remote-NUMA load latency 贡献约 31%，并带有对应采样误差”比“X 使用了 31% 内存带宽”更符合证据。

**与已有工作的差别。** heap profile 主要按 allocated bytes 加权，DAMON 按访问行为加权地址区域，`perf mem` 按采样访问观察硬件成本。这里新增的是保留来源和覆盖语义的跨层融合。

**可实现产物。** DAMON adapter、perf-event/BPF sampling adapter，以及一个可以在 reserved bytes、resident bytes、访问频率、采样延迟和 NUMA locality 之间切换的 profiler 视图。

**评测。** 使用 STREAM 类带宽 workload、random pointer chasing、冷热缓存混合和 NUMA placement workload。与已知地址区间的生成器以及适用的硬件计数器做对照，改变采样率和 DAMON region 上限，画出 accuracy-overhead 曲线。

**学术价值。** 可以量化“加入访问采样以后，内存归因是否真的发生足够大的变化”。

**生产价值。** 能区分“大但冷的缓存”和“体积较小却真正制造远程内存或带宽压力的结构”。

**失败条件。** 如果采样偏差或硬件缺失让不同机器上的结果不稳定，系统应该退回区域级 working-set 分析，而不是宣称可移植的带宽归因。

### 3. 内存归因的 ground-truth benchmark

**缺口。** 内存 profiler 的评测常看运行开销，或者已知 leak/hot region 是否出现在报告里，这不足以比较分配、常驻、访问、回收和迁移之间的来源准确性。

**机制。** 构造能暴露真实“逻辑资源 -> 虚拟区间 -> 可控页面活动”的 workload。测试 harness 私下记录 oracle，profiler 只能使用正常观测接口。案例至少覆盖 untouched reservation、sparse touch、free/reuse、共享文件映射、`fork` + COW、THP split/collapse、reclaim/refault、NUMA migration，以及多个逻辑资源共用一个 allocator arena。

**可实现产物。** 一套可复现 benchmark 和 trace corpus，定义 allocation identity、page-state transition 和 attributed cost 的统一结果 schema。

**评测。** 在存在精确 ground truth 的地方计算 byte/page-level precision 和 recall；对采样访问计算估计误差和置信区间覆盖。额外注入 event loss 并跨内核版本运行。最简单的 baseline 应该包括 allocation-only tracing、`smaps`、Idle Page Tracking 和 DAMON。

**学术价值。** 它把“更好的内存归因”从可视化印象变成可以被证伪的系统性质。

**生产价值。** profiler 可以明确说明哪些 workload 能精确归因，哪些只能退化到估计。

**失败条件。** 如果 benchmark 显示逐页 lineage 相比 VMA-level provenance + DAMON 几乎不增加诊断准确率，后续工作应该优化更便宜的 VMA-level 方案。

## 第一版不应该一直追踪每一个页面

最直接的设计会记录每次分配、每次缺页、每次回收、每次迁移和每个内存样本。这样很容易让 profiler 自己成为新的观测开销来源。

更现实的第一版应该持续保留粗粒度语义身份，只在必要时花逐页预算：

1. 持续追踪 allocation 和 mapping generation，因为它们定义来源拓扑；
2. 用 `smaps`、DAMON 或内存压力信号找出值得深入分析的区域；
3. 对这些区域或一个有限诊断窗口开启更细的 page/fault/access 采集；
4. 在结果里保留 lost-event 和 sampling metadata；
5. 区域变冷或诊断结束后回收细粒度状态。

这个策略与前面的 [异步 eBPF 因果分析报告](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/) 使用同一个思路：定义拓扑的证据和昂贵上下文应该使用不同预算。这里的拓扑是 allocation-to-region identity，逐页活动才是昂贵细节。

## 哪些结果会改变这个判断？

最强的反例是 Linux 现有的 VMA 粒度已经够用。如果把 allocator stack 关联到 VMA，再结合 DAMON region 统计，就能和逐页 lineage 一样解决真实内存事故，那么逐页来源关联只是增加复杂度。

另一类反例是主要成本来自 page cache 或全局共享内存。这时可能根本没有一个有意义的用户态 allocation owner。更合适的身份可能是 inode、memcg、共享内存对象或其他资源。通用系统必须允许这些身份，而不是强迫所有成本归到 heap stack。

最后，硬件访问采样可能过于依赖平台，无法成为可移植的带宽归因基础。即使如此，如果 provenance ledger 能解释 reserved、resident、touched、reclaimed 和 migrated memory，核心设计仍然有价值；带宽采样则应该保持为带支持矩阵的可选证据。

只有 ground-truth benchmark 证明跨层关联能够显著改变故障诊断或优化决策，而且收益足以覆盖观测成本，这套设计才值得做成持续运行的系统。在此之前，更合理的实现顺序是先保留 allocation 和 mapping identity，再按可用性附加逐页证据，并始终把不确定性留在输出里。
