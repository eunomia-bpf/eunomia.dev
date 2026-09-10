---
date: 2026-09-10
title: "一个 eBPF 操作跨主机和硬件拆分后，还能保持同一种语义吗？"
description: "高级 eBPF 卸载会把同一个策略效果拆到主机与硬件之间。本文分析如何在部分完成、重试、回退和设备故障时保持原子性、顺序、状态一致性与可恢复语义。"
tags:
  - Daily Report
  - eBPF
  - SmartNIC
  - DPU
  - Offload
  - Distributed Systems
  - Verification
research_question: "一个高层 eBPF 语义操作被拆分到内核、NIC/DPU 或其他硬件执行后，怎样才能在部分完成、重试、乱序、回退和设备故障时仍保持同一个可观察结果？"
source_cutoff: 2026-09-10
status: daily-report
---

# 一个 eBPF 操作跨主机和硬件拆分后，还能保持同一种语义吗？

设想一个 eBPF 策略收到数据包后，先检查主机维护的租户额度，再让 SmartNIC 做变换或重定向，最后更新记账状态。整个流程都留在 CPU 上最容易推理，但可能浪费硬件快路径；把整个程序都搬到 NIC 上又未必可行，因为权威额度、权限检查或某个必要操作仍然只存在于主机。

很自然的折中方案是把一个操作拆开：设备负责本地而且便宜的部分，主机负责依赖主机状态或权限的部分。性能可能很好，但随之出现一个新的正确性问题：如果设备已经完成自己的那一半，而主机还没有提交；或者超时重试后其中一侧又执行了一遍，系统最终对外到底算执行了一次、两次，还是只执行了一半？

Linux 和现有研究已经包含这个问题的若干局部答案。XDP 可以把 frame 交给另一颗 CPU 或另一个设备相关程序；Linux 能把 BPF 程序绑定到硬件 offload 设备；网络子系统会把一个高级功能拆到软件和硬件之间；fabric_ext 这类系统甚至会把一个策略编译到多个异构目标。真正缺少的是一个通用的 eBPF 合约，用来描述**同一个逻辑效果跨多个执行域实现时的提交语义**：什么时候算完成，哪些效果可以重放，谁拥有权威状态，以及两侧如何确认自己处理的是同一个策略 generation。

<!-- more -->

这和此前的[异构 eBPF 执行放置](https://eunomia.dev/zh/research/heterogeneous-ebpf-execution-placement/)不同。那个问题是在多个合法目标中决定“程序应该运行在哪里”。它也不同于[架构相关的 eBPF specialization](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)：后者关心某个 portable operation 在当前机器上有没有可用实现；最近的[原生操作信任边界](https://eunomia.dev/zh/research/ebpf-native-operation-trust-boundary/)则关心 delegation 以后哪些 native code 必须被信任。本文假设目标已经选好，也已经满足基本的能力与信任要求。剩下的问题是：**一个 operation 只在多个域里分别完成一部分时，怎样避免产生对外可见的“半个结果”。**

## BPF 已经有显式 handoff，但还没有通用的拆分操作合约

Linux XDP 是很好的起点，因为它不会把 handoff 伪装成普通的本地执行。`BPF_MAP_TYPE_CPUMAP` 可以把 `xdp_frame` 重定向到另一颗 CPU，并在那里运行第二个 XDP 程序。`BPF_MAP_TYPE_DEVMAP` 可以给目标设备关联另一个 XDP 程序，这个程序会在 `XDP_REDIRECT` 之后、frame 进入发送队列之前执行。换句话说，handoff 本身就是 API 的一部分。

这种机制比“编译器悄悄把程序切成两半”容易推理。第一个程序通过明确的 XDP action 结束，frame 的所有权在一个已知边界上发生变化，下一阶段也在文档定义的 context 中运行。不过，应用层面的事务语义仍然需要自己处理。假如第一阶段增加了 quota counter，第二阶段却丢弃、重复或重新路由这个 frame，内核不会自动把两个效果包装成一个跨阶段 transaction。

Linux 当前的 BPF 硬件 offload 路径也有类似边界。`kernel/bpf/offload.c` 把 device-bound program 限定在 XDP 和 `SCHED_CLS`，将程序关联到具体网卡，并允许设备后端参与 verifier 的准备、逐指令检查与 finalize。它定义了真实的 BPF offload 合约，但这个合约主要回答“**哪个 program 绑定到哪个 device**”。它并没有一个通用表示方式，说明前 40 条指令和一次 host map 更新属于同一个操作的主机阶段，后 40 条指令和一次 packet mutation 属于它的设备阶段。

因此，whole-program equivalence 和 split-operation correctness 是两种不同的性质。整个程序被设备实现时，可以拿最终行为与原始 BPF 比较；一旦实现被切开，正确性还取决于两个部分之间的协议。

## Linux 里的其他 offload 已经把协议问题暴露得很清楚

更广泛的 Linux 网络栈里，软件和硬件共同实现一个高级操作早已存在，这些接口很适合作为反例。

XFRM device 接口区分两种 IPsec offload。**Crypto offload** 只把加密和解密交给 NIC，其余工作仍由内核完成；**packet offload** 则进一步让 NIC 负责封装，并要求内核和 NIC 的 SA 与 policy 状态保持同步。两者的区别不仅是“搬走了多少工作”。后一种模式让硬件参与了更多逻辑操作，因此也需要更强的共享状态协议。

Netfilter flowtable 的硬件 offload 展示了另一种问题。内核文档明确说明，把 flow 加入硬件是异步的，所以规则真正装进设备之前，少量数据包仍可能经过软件 fast path；文档还提醒，转发信息变化时硬件里的 flow state 可能变旧。这意味着同一个逻辑流量类别在一段时间内会同时存在软件和硬件路径。

这些都不是 eBPF API，但它们说明“已经打开 hardware offload”绝不等于“语义天然原子”。正确性取决于哪些状态被同步、什么时候切换 ownership、哪些请求仍能走旧路径，以及 stale 或失效的硬件状态如何被修复。

只要高级 eBPF delegation 开始拆分一个效果，而不是整体搬走一个自包含程序，同样的问题就会出现。

## 研究系统已经证明跨设备分解是有价值的

[hXDP](https://www.usenix.org/conference/osdi20/presentation/brunella) 证明了真实 XDP 程序可以运行在 FPGA NIC 上。它不仅实现了 BPF 指令执行，还提供优化编译器、扩展 BPF ISA，以及 FPGA 版本的 XDP maps 和 helper functions。论文中的实现使用大约 15% FPGA 资源，在评测环境里达到接近高端 CPU core 的 packet-processing throughput，并报告约 10 倍更低的 forwarding latency。对本文最重要的不是峰值数字，而是一个事实：要保留熟悉的 XDP 行为，设备必须重现的不只是 BPF 算术，还包括周围的 map 和 helper 环境。

更近期的 [fabric_ext](https://arxiv.org/abs/2607.26335) 已经走到 whole-program offload 之外。它用 semantic movement graph 表示 source、destination、ordering、ownership，以及 move、checksum、filter、reduce、replicate、persist 等 transformation，再把一个策略 lower 成 GPU、driver/runtime、DPU/NIC 与 CXL 侧的 BPF program、verifier obligations、不同 consistency class 的 maps 和后端 artifact。

这说明更高层的跨设备分解确实值得研究，也把剩余边界变得更清楚。编译器可以知道某个 operation 是 `Reduce` 后接 `Persist`，也可以知道两个阶段共享 ownership edge；但在 production recovery 里，如果一个目标已经看到完成，而另一个目标 reset、retry，或者仍运行旧 generation，系统还需要明确答案。

因此下一步应该比“通用分布式事务”小得多，但又比 best-effort handoff 强。我们真正需要描述的是：**哪些高级 operation 可以被拆，以及拆开后的 commit semantics 是什么。**

## 拆分后的多个阶段必须仍然产生一个对外结果

考虑一个 `charge_and_redirect(tenant, packet)` 操作，它希望产生两个效果：

1. 从权威 tenant budget 里消耗一个额度；
2. 把数据包发送到选定设备路径。

它可以有多种拆法。主机可以先 reserve 额度，再给 NIC 一个只能消费一次的发送 capability；NIC 也可以先做 tentative processing，再请求主机 commit；如果整个 operation 被设计成 replay-safe，两边还可以用同一个 identity 做 retry。

真正不能含糊的是：**什么时候这个操作对外算提交。**

一个有用的 split-operation descriptor 至少要说明：

- 稳定的 operation type 和 policy generation；
- retry 后仍保持不变的 operation-instance identity；
- 每个被读取或修改的 state object 的权威 owner；
- commit 之前允许发生哪些效果，哪个效果真正构成 commit；
- 各阶段是否 idempotent、commutative、compensatable，或者绝不能重复；
- host 与 device effect 之间要求什么 ordering；
- timeout、device reset、generation change 和 duplicate completion 后的终态规则。

这并不意味着每个 packet 都要跑 two-phase commit。很多 fast path 可以用便宜得多的协议：主机签发一个有界的一次性 capability；设备用 operation ID 做幂等变换；可交换的 counter update 延迟合并。重点不是强迫所有工作使用同一种事务，而是让编译器和 runtime 有足够信息判断一个 split 是否安全。

第一版原型甚至不必先改 Linux ABI。可以在 [bpftime](https://github.com/eunomia-bpf/bpftime) 这类 runtime 上实现 host-side BPF stage、软件模拟的 device stage 与 fault injection，先验证合约能不能发现真实错误，再决定哪些字段值得进入更低层接口。一个可信的研究结果应该先证明这套合约的价值，而不是先宣称任意 eBPF 都能自动分布式执行。

## 现有研究还缺什么

### 拆分语义仍然散落在各个子系统里

CPUMAP 和 DEVMAP 各自定义了 XDP handoff；XFRM 有自己的软件/硬件分工和状态同步；flowtable offload 又有另一套 transition model；硬件 BPF offload 提供 device-specific verifier 与 translation hooks。这些接口能工作，是因为每个 subsystem 都在本地编码自己的假设。

缺少的是一套可复用的 eBPF 层 vocabulary，用来描述一个高级 operation 横跨两个执行域时的 effect semantics。否则每一个新 compiler 或 accelerator 都要重新决定：handoff 前允许发生什么、什么状态必须保持权威、哪些 retry 可以安全执行。

### Verifier safety 结束在跨域 commit semantics 之前

BPF verifier 可以证明 program 的内存与类型安全，也能检查声明过的 call constraint；设备 backend 还可以进一步验证或翻译 offloaded program。但这些结果都不能证明“主机 reserve 一次 + 设备 packet effect 一次”会按照要求组合成恰好一个逻辑操作。

这里缺的不是另一条 instruction safety rule，而是多个已验证 stage 之间的 protocol property。

### Offload evaluation 很少主动制造 failure 和 generation change

Throughput 与 latency 是 accelerator 最自然的指标，但它们通常看不到真正会让 split 出错的场景：timeout 后设备重复 completion；主机已经升级 policy，而旧 device rule 还在执行；host state mutation 已成功但设备 reset；fallback 又把同一个 effect 做了一遍。

评价 split-operation mechanism 时，这些事件应该是 benchmark 的输入，而不是偶尔发生的事故。否则 steady state 看起来完全等价，真正切换或故障时却可能分叉。

### 哪一类 operation 值得拆，目前没有足够证据

完全通用的协议可能太贵。Stateless transform、monotonic counter、bounded reservation、idempotent write，也许已经覆盖了大部分值得 offload 的操作，并且只需要很简单的规则。反过来，把不可逆 external effect 和强一致 host state 混在一起的操作，也许根本就应该留在一个执行域里。

在标准化跨设备 transaction abstraction 之前，系统研究首先要回答这个边界到底在哪里。

## 兼具学术价值与生产价值的方向

### 1. 用显式 effect class 编译可拆分操作

**Gap。** 现有 BPF type 能表达 value 和 verifier-visible safety，但 compiler 要决定是否 partition 一个高级 operation 时，没有机器可读的 retry 与 commit 语义。

**Mechanism。** 给高级 operation 的 effect 标上少量明确类别，例如 pure、idempotent、commutative、reservable、compensatable、non-repeatable，同时声明 authoritative state 与 commit effect。Compiler 只有在能为这些 effect class 选择合法协议时才允许 split。例如 host quota 可以 reserve，再发一个 one-use device capability，这不需要通用 transaction coordinator；而“先做不可重复 external write，再补 host bookkeeping”可以直接判成不可拆。

**Delta。** Architecture capability manifest 解决的是“这个 target 能不能运行某个 implementation”；native-operation contract 约束的是“一个 trusted implementation 可以产生什么效果”。这里解决的是第三个问题：**多个 implementation 能不能在 failure 下共同实现一个 logical effect。**

**Artifact。** 一套 descriptor format、compiler pass 和 runtime library，先实现三到四种具体协议，并接到 host BPF 与软件 SmartNIC/DPU emulator 上。

**Evaluation。** 选择 packet transformation、quota enforcement、telemetry aggregation、movement/checksum pipeline，与 whole-host、可用时的 whole-device、手写 ad hoc split、effect-typed split 比较。主动注入 retry 和 device reset，测错误 external outcome、duplicate/lost effect、runtime overhead、额外 latency，以及 compiler 能安全接受多少 split。若普通的显式 stage API 能用明显更少的 metadata 与机制获得同样 fault coverage，这个方向就不值得继续。

### 2. 在 handoff 上加入 generation-bound operation receipt

**Gap。** Timeout 本身无法告诉主机：设备完全没看到操作、已经完成，还是在旧 policy generation 下完成。

**Mechanism。** 每个 split instance 带一个紧凑 identity，例如 `(policy_generation, operation_id, stage)`。设备只保存该 effect class 做 retry 或 dedup 所需的最小状态；completion 返回包含 generation、stage outcome 与 effect identity 的 receipt。主机用同一个 operation ID 重试时可以去重，旧 generation 的 completion 也不能悄悄满足新操作。

对超高速路径，不需要把每个 receipt 都导出给 userspace。它可以只存在于 bounded device state、compact ring 或 sampled audit mode 中。保证强度应该随 effect 严重程度变化，而不是把每个 packet 都变成分布式日志记录。

**Artifact。** 一套 host/device handoff library 与 fault-injection hooks，能模拟 delayed completion、duplicate completion、device reboot 和 stale-generation state。

**Evaluation。** 比较无 identity、只有 generation identity、完整 operation identity 三种方案，测 duplicate effect、stale-generation acceptance、memory/lookup cost、recovery latency 与最大 sustainable operation rate。Ablation 要直接说明 operation-level identity 相比已有 generation fence 是否真有额外价值。

**Academic value。** 一般化问题是：高吞吐 programmable datapath 需要保存多少跨域状态，才能获得 replay-safe semantics。

**Production value。** 设备 reset 或 timeout 后，operator 可以按照有界规则恢复，不再只能在 blind replay 与丢弃 uncertain work 之间二选一。

### 3. 为 partial offload 建一个 semantic fault benchmark

**Gap。** 当前 offload benchmark 很容易证明“某个 target 更快”，却很少证明 split 在 transition 与 failure 时仍产生同一个结果。

**Mechanism。** 每个 workload 先定义 observable semantic oracle，再分别用 pure host、pure device、split 三种配置运行。Fault 精确注入到 handoff 周围：device acceptance 前、device effect 后但 completion 前、host state mutation 后、policy generation replacement 中，以及 fallback activation 时。

**Artifact。** 一个可重复 benchmark，包含 packet traces、state snapshots、fault schedules，以及能检测 duplicate、lost、reordered 与 stale-generation effect 的 oracle。Linux XDP multi-stage path 可以作为容易复现的 baseline，再加入至少一种真实 SmartNIC/DPU 或高保真 emulator 路径。

**Evaluation。** 第一指标先报告 semantic divergence rate，再看 throughput、p99 latency、recovery time、receipt/state memory 与 host-device traffic。比较 best-effort split、stop-the-world handoff、generation fencing 和 effect-aware replay。必须加入一个 stateless independent packet workload，让最简单方案在本来就不需要复杂协议时能够赢，否则 benchmark 只会机械奖励更多机制。

**Academic value。** 这可以给 partial offload 的 correctness/performance frontier 一个共同测量方法。

**Production value。** Driver、compiler partitioner 和 programmable-NIC runtime 可以把它变成启用新 split mode 之前的 release gate。

## 哪些结果会改变这个判断？

如果实际 eBPF offload 很少需要拆分一个逻辑效果，而 whole-program placement、显式 XDP stage boundary 或 stateless device function 已经能覆盖大部分有价值部署，那么通用 split-operation contract 的必要性会明显下降。

另一种可能是，现有 subsystem-specific protocol 本身已经形成足够小的共同接口，不需要新的 BPF semantics。XFRM 风格的状态同步、generation fencing，再加普通的幂等 request ID，也许只要 runtime 组合得当就够了。

最强的反证应该来自 fault injection：如果 effect typing 和 operation receipt 相比简单的 explicit-stage baseline 没有多发现任何外部可见错误，或者它们制造的 state traffic 足以吃掉 offload 的性能收益，那么这层 abstraction 就应该删掉。

在看到这些证据之前，高层 eBPF offload 最好明确区分两件事：**搬走一个 implementation** 和 **拆开一个 operation**。前者需要 capability、equivalence 和 trust evidence；后者还需要 commit 与 replay contract，因为两个分别正确的 stage 仍然可能组合出一个错误的外部结果。

## References

- Linux kernel source, [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c), accessed 2026-09-10.
- Linux kernel documentation, [`BPF_MAP_TYPE_CPUMAP`](https://docs.kernel.org/bpf/map_cpumap.html), accessed 2026-09-10.
- Linux kernel documentation, [`BPF_MAP_TYPE_DEVMAP` and `BPF_MAP_TYPE_DEVMAP_HASH`](https://docs.kernel.org/bpf/map_devmap.html), accessed 2026-09-10.
- Linux kernel documentation, [XFRM device: offloading IPsec computations](https://docs.kernel.org/networking/xfrm_device.html), accessed 2026-09-10.
- Linux kernel documentation, [Netfilter flowtable infrastructure](https://docs.kernel.org/networking/nf_flowtable.html), accessed 2026-09-10.
- Marco Spaziani Brunella et al., [hXDP: Efficient Software Packet Processing on FPGA NICs](https://www.usenix.org/conference/osdi20/presentation/brunella), OSDI 2020.
- Yiwei Yang and Andi Quinn, [The Fabric Is the Cluster Driver: Cross-Layer eBPF Policies for GPU-CXL Fabrics](https://arxiv.org/abs/2607.26335), arXiv:2607.26335, 2026.
