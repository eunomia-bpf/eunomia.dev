---
date: 2026-09-22
slug: ebpf-kernel-interface-negotiation
title: "eBPF 加载器能把 kfunc 简化成“有”或“没有”吗？"
description: "kfunc、struct_ops 与 BPF iterator 都带有类型、上下文和 provider 约束，单纯检查接口是否存在不足以安全选择程序变体。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - kfunc
  - struct_ops
  - Compatibility
research_question: "当不稳定或与 provider 绑定的 BPF 内核接口持续演进时，加载器在选择程序变体前究竟应该协商哪些条件？"
source_cutoff: 2026-09-22
status: daily-report
---

# eBPF 加载器能把 kfunc 简化成“有”或“没有”吗？

一个 eBPF 加载器查看目标内核的 BTF，发现自己需要的 kfunc 名字存在，于是选择更快的新版本程序。相比直接判断 `uname -r`，这已经合理得多。

问题是，名字只描述了接口契约的一小部分。某个 kfunc 可能只对一种 BPF program type 注册；它的指针参数可能受到 ownership、nullability、RCU、锁或生命周期约束；open-coded iterator 实际上是一组构造、取下一项、析构的协议，连 state struct 的大小都属于用户可见 API；XDP metadata kfunc 即使在内核里存在，具体网卡驱动仍可能在运行时返回 `-EOPNOTSUPP`。

因此真正需要回答的不是“这个接口有没有”，而是：**目标内核、BPF program type、provider 和 verifier 是否共同满足这份 BPF artifact 所依赖的那一个具体接口契约。**

本文接在此前的[基于 capability evidence 的准入](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)和[跨内核语义兼容](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)之后。前两篇分别讨论“这个 host 能不能接受 artifact”和“接受之后应用语义是否仍然成立”。这次关注更窄的一层：**在真正进入 verifier admission 之前，加载器怎样选择正确的接口变体。**

<!-- more -->

## kfunc 的名字只是兼容性的一个维度

Linux 明确把 kfunc 和传统 BPF helper 放在不同的稳定性边界上。当前内核文档说明，kfunc 不提供稳定接口保证，可以随内核版本改变；它的可见性可以按 BPF program type 注册，verifier 还会结合 BTF 类型与 `KF_ACQUIRE`、`KF_RELEASE`、`KF_RET_NULL` 等语义标记检查调用。

因此，下面几种完全不同的状态，初看都可能被简化成“这个函数存在”：

```text
名字不存在
    -> 当前变体显然不能调用

名字和 BTF signature 存在，但 program type 不匹配
    -> verifier 不允许当前程序调用

名字、signature、program type 都存在，但 verifier contract 不同
    -> pointer、lifetime 或调用上下文假设可能已经不成立

调用能够通过 verifier，但 provider 没实现具体能力
    -> 运行时对当前 device/context 返回 unsupported
```

最后一种并不是理论情况。Linux 的 XDP RX metadata API 提供 timestamp、RSS hash、VLAN 等一组 kfunc，由网卡驱动选择是否实现。文档明确规定：驱动没有实现某个 metadata operation 时返回 `-EOPNOTSUPP`，并且可以通过 netlink 查询某个 netdev 支持哪些 metadata kfunc。

所以，把 `has_bpf_xdp_metadata_rx_timestamp=true` 记成一个 host-global Boolean，会丢掉最重要的 scope。真正有意义的事实更接近：“这个 XDP 程序在这块 netdev 上，可以按照这组参数和返回值规则使用该 operation。”

## 越新的 BPF 接口越像协议，而不是单独一个函数

Open-coded BPF iterator 把这个问题展示得尤其清楚。Linux 文档把一个 iterator 定义成一组紧密绑定的 kfunc：constructor、`next`、destructor，再加一个 iterator-specific state struct。Verifier 会保护这块 state，并依赖 `next` 最终返回 `NULL` 的协议。文档还特别指出，state struct 的大小本身就是 user-visible API，改变它会破坏 backwards compatibility。

也就是说，iterator 的兼容关系至少同时包含：

```text
iterator state type 和 size
        + constructor signature 与初始化规则
        + next signature、返回类型、nullability 与 verifier 语义
        + destructor signature 与 lifetime 规则
```

只检查 `bpf_iter_<type>_next` 这个名字是否存在，根本无法证明整组关系仍然成立。

`struct_ops` 又提供了另一种同类问题。以 `sched_ext` 为例，用户态通过 `struct sched_ext_ops` 加载 BPF scheduler；当前文档说明除了 `ops.name` 外，其余 operation 都是 optional，没有实现的 callback 还可能走默认行为。加载器如果要在多个 scheduler 变体之间选择，就需要知道目标 `struct_ops` 的 schema、当前变体真正依赖哪些 callback 和 flag，而不仅是 `CONFIG_SCHED_CLASS_EXT` 有没有打开。

这些接口有价值，恰恰因为它们能跟着内核 subsystem 一起快速演进。强行把它们当成永远冻结的 helper 会牺牲这种灵活性。更合适的方向是承认接口会移动，并把应用依赖的那部分契约显式化。

## 直接试着加载很权威，但它不是完整的 negotiation protocol

最有力的简单方案其实很直接：编译几份 BPF object，按优先级逐个 load，让 verifier 拒绝不兼容的版本。这个办法有明显优点。目标内核的 verifier 才是安全性和 admission 的最终权威，用户态没有必要复制所有 verifier 规则。

如果应用只有两份明确的 variant，而且部署环境简单，这可能就是最好的设计。

问题出现在 interface family 和 fleet 扩大之后。一次 load failure 可能同时混入多种原因：BTF type 缺失、program type 不允许、kfunc signature 变化、ownership contract 不匹配、`struct_ops` field 不支持，或者完全无关的 verifier constraint。顺序试加载可以告诉你“最后哪一份成功了”，却不一定能解释“究竟是哪条接口要求迫使系统退回这一份”。Provider-specific 能力甚至可能要到程序实际运行后才暴露。

而生产系统需要回答更具体的问题：

- 为什么这个 node 选择 fallback object？
- 同一台机器换到另一块 netdev，是否应该选择另一份 variant？
- 哪一个 kernel-interface change 要求我们重跑 compatibility matrix？
- 当前失败是预期的接口不匹配、应用 bug，还是 verifier regression？

因此，trial load 应当继续作为最终 authority，但它更适合作为 negotiation path 的最后一道检查，而不是唯一的兼容性表示。

## 现有研究还缺什么

### 接口要求散落在 BTF、verifier metadata、文档和 provider state 里

BTF 已经提供丰富的类型信息，内核也会记录 kfunc visibility 与语义 flag；某些 subsystem 还暴露额外状态，例如 XDP RX metadata 对每个 netdev 的支持情况。但应用 artifact 没有一个通用、machine-readable 的方式声明：自己究竟依赖其中哪一部分。

加载器当然可以把这些判断全部硬编码进去，但这样 compatibility policy 就退化成应用内部逻辑。缺少的是一份 artifact-level requirement description，让加载器在真正执行昂贵或可能带副作用的部署步骤之前，先把 artifact requirement 与 target evidence 做结构化匹配。

一个有区分力的实验是：同一份 requirement description 能否在多个 upstream kernel、distribution kernel、program type 和 device 上正确分类 object，而不需要写 kernel version range。

### Host-level capability probe 没有描述接口在哪个上下文才有效

前一篇 [capability evidence](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/) 报告主张直接探测真实 host，而不是相信版本号。但 probe 本身仍然可能 scope 太宽。一个 kfunc 对某种 BPF program type 合法，对另一种却不合法；同一台机器上的两块网卡，也可能对 XDP metadata 提供不同支持。

因此缺少的是 **scoped capability evidence**：证据必须绑定到真正决定接口可用性的 context。对不同 API，这个 scope 可能包含 program type、device、module/provider、sleepability、attachment type，或其他 verifier-visible condition。

Benchmark 可以故意构造“host-global feature 看起来完全相同，但合法 context 不同”的机器，然后测量全局 feature probe 造成多少 false admission。

### Compatibility CI 不知道一次接口变化应该触发哪些测试

大规模 kernel matrix 可以抓到问题，但成本高，而且每个 cell 证明什么往往不清楚。某个 kfunc signature、iterator state type 或 `struct_ops` field 改变后，并不是所有 BPF 应用都需要重跑所有组合；真正依赖该 contract 的 artifact 才需要重点验证。

缺少的是一种 dependency-aware 的连接方式：把 kernel-interface delta 映射到 artifact requirement，再选择最小但有意义的 compatibility matrix。没有这层连接，项目要么对 unstable interface 测得不够，要么持续运行很大的 matrix，却解释不了每个测试 cell 的必要性。

最直接的验证方式是 historical replay：给定一串内核接口变化，dependency model 是否能选出足以抓住真实 compatibility failure 的测试，同时明显减少无关 cell。

## 兼具学术价值与生产价值的方向

### 方向一：给每个 BPF artifact variant 带上 typed interface requirement

**缺口。** BPF object 已经可以包含 BTF 与 CO-RE metadata，但更高层的接口假设通常仍然散落在 loader source code 和 variant-selection logic 里。

**机制。** 为每个程序变体附加一份紧凑 requirement manifest，描述 artifact 真正依赖的接口属性，例如：

```text
kfunc:
  name: bpf_example
  program_type: tracing
  signature: <BTF-derived type identity>
  effects: [acquire, nullable-return]

iterator:
  state_type: bpf_iter_example
  state_size: ...
  protocol: [new, next, destroy]

provider_feature:
  scope: netdev
  operation: rx_timestamp
```

加载器从 BTF、内核公开的 feature state 与 provider query 生成 target profile，先做 structural match，再尝试加载 object。Verifier 仍然保留最终 authority；manifest 不需要、也不应该重新实现 verifier。

**与现有机制的差异。** CO-RE 描述“怎样根据 target type relocation”。这里的 manifest 描述“这一份 object variant 在 relocation 与 verification 之前究竟要求哪一种接口 contract”。

**可实现 artifact。** 给 libbpf 增加 manifest generator 与 resolver，再给 `bpftool` 增加一个视图，可以同时打印 artifact requirement 与命中的 target evidence。

**评测。** 覆盖多个 upstream/distribution kernel、program type、kfunc family、iterator、`struct_ops` 与 XDP device，比较 kernel-version gate、只查 BTF name、trial-load-only 与 typed matching。指标包括 false admit、false reject、失败的 trial load 数量，以及诊断结果能否精确指出哪条 requirement 不满足。

**学术价值。** 可以研究一个持续变化的 kernel interface 到底有多少部分能被 declarative contract 描述，而无需复制 verifier semantics。

**生产价值。** 一套应用携带多份 BPF object 时，variant selection 变成可解释、可测试的流程，而不是连续尝试直到某份程序碰巧通过。

**失败条件。** 如果 typed matching 不能比简单 ordered trial loading 更准确地预测可用 variant，或者 manifest 最终需要人工复制 verifier internals，这一层就不值得存在。

### 方向二：按 scoped capability 协商 variant，而不是维护一个 host feature bitmap

**缺口。** Machine-wide capability map 会丢掉 program type、device、module 或 attach context 等真正决定接口合法性的 scope。

**机制。** 把 variant selection 表示成一次有上下文的 constrained matching：

```text
artifact requirements
    x kernel/BTF generation
    x BPF program type
    x attach target
    x provider/device identity
    x verifier-relevant context
        -> selected variant + rejected alternatives + evidence
```

系统输出一份小型 negotiation receipt，记录最终选择的 variant，以及让它成立的 scoped fact。Cache 也只能复用到这些事实仍然有效的 scope：netdev capability 不能升级成 host-global fact；某个 program type 的 kfunc registration 也不能拿去证明另一个 program type。

**与 9 月 15 日报告的差异。** 之前的 artifact-bound receipt 解释“为什么这份 artifact 被 host 接受或拒绝”。这里发生得更早也更具体：当一个应用支持多种 unstable interface shape 时，它决定“哪一份 artifact 应该进入 admission”。

**可实现 artifact。** 一个 resolver library、一套 normalized scoped-capability schema，以及 kfunc、iterator、`struct_ops` 与 device-specific feature discovery fixture。

**评测。** 构造 kernel version 和 BTF 相同，但 device support 或 program-type eligibility 不同的 host。测量错误 variant choice、fallback rate、probe cost 与 cache invalidation error，并加入 reboot、module reload、device replacement 和 mixed-netdev 场景。

**学术价值。** 可以验证 capability negotiation 是否应该建模成 context-dependent relation，而不是 flat feature set。

**生产价值。** Operator 可以直接回答“为什么这台 node 的这块 device 选了这个 variant”，而且只需要让 scope 已经改变的 receipt 失效。

**失败条件。** 如果真实部署中的绝大多数 BPF-facing capability 都确实是 host-global，这种 scoped negotiation 会增加很多复杂度，却避免不了多少故障。

### 方向三：根据 interface diff 与 artifact dependency 自动生成 compatibility CI

**缺口。** Kernel matrix 很贵，静态 support table 又会快速过期；两者都没有直接表达“这次接口变化究竟影响哪些已发布 artifact”。

**机制。** 从 requirement manifest 建立每份 artifact 的 interface dependency。面对 candidate kernel，计算与这些 dependency 相关的 normalized delta，包括 BTF signature、kfunc visibility/effect、iterator state/protocol、`struct_ops` schema 和 provider feature inventory。然后用 dependency graph 选择能够覆盖所有已变化 requirement 的最小 artifact/kernel/provider 组合。

这并不替代周期性的 broad testing，而是在每次 kernel update 上增加一层便宜、可解释的 targeted test。

**可实现 artifact。** Interface-diff tool、dependency graph 与 CI planner。Planner 输出可重现的 test matrix，并把每个 cell 链接回触发它的 changed contract。

**评测。** Replay 历史 kernel release，再加入可控 synthetic change。比较 full Cartesian matrix、固定 LTS/current sampling 与 dependency-selected matrix。指标包括抓到多少 compatibility failure、matrix size、诊断时间，以及是否漏掉“单个 dependency 看起来没变，但组合之后出错”的 interaction。

**学术价值。** 这是一个在 dependency information 不完整条件下，为持续变化的 typed kernel/application boundary 选择测试集合的问题。

**生产价值。** eBPF 项目可以把 CI budget 花在实际 shipped artifact 消耗的接口上，并在 unstable dependency 变化时更早得到解释明确的 warning。

**失败条件。** 如果真实 failure 主要来自 dependency graph 之外的 verifier behavior 或 cross-subsystem interaction，缩小 matrix 会制造危险 blind spot，这个 planner 就只能作为建议工具。

## 实际 loader 仍然应该让 verifier 拥有最终决定权

这里并不是要把 verifier 搬到用户态。一个实际的 production path 可以保持简单：

```text
收集 scoped target evidence
        -> 排除明显不兼容的 variant
        -> 选择最匹配的 artifact
        -> 执行 CO-RE relocation
        -> 由目标内核 verifier 做最终 admission
        -> attach，并在必要时验证 provider/runtime behavior
        -> 记录 negotiation 与 admission evidence
```

这种分层方式保留了内核的权威，同时让 verifier 前后发生的 compatibility failure 更容易解释。它也不需要假装 unstable interface 永远稳定。加载器只要知道自己依赖 contract 的哪些部分，并在对应 scope 变化后重新评估即可。

只使用稳定 helper 和稳定 program type 的应用，manifest 可以非常小甚至为空。只有那些已经因为接口演进或 provider scope 而需要维护多条代码路径的项目，才应该承担这套机制的成本。

## 哪些结果会改变这个判断？

如果一次覆盖广泛的实证研究发现，ordered trial loading 已经能处理绝大多数 unstable BPF interface，而且 ambiguous failure 极少、启动成本可以忽略、诊断信息也足够，那么单独增加 negotiation layer 很可能只是在重复 verifier。

如果 kfunc、iterator 与 `struct_ops` 在实践中逐渐形成长期稳定的 contract，而 provider-specific capability difference 也越来越少，显式接口契约的收益同样会下降。这里的价值前提本来就是接口确实在持续移动。

最后，dependency-driven CI 只有在 interface change 真能预测 application risk 时才有意义。如果历史 failure 主要由 requirement graph 捕捉不到的 verifier behavior 或跨 subsystem interaction 造成，那么小型 targeted matrix 会比 broad testing 更危险。

目前 Linux 提供的接口边界仍然更复杂：kfunc 明确没有 hard stability guarantee，program-type visibility 是调用 contract 的一部分，open-coded iterator 同时暴露 protocol 和 state layout，而 XDP metadata 的支持范围还能随 device 变化。更合适的 deployment contract 因此应该是 **typed、scoped，并最终由 verifier 背书**，而不是把一个接口压缩成简单的“有”或“没有”。

## 参考资料

- [Linux 内核文档：BPF Kernel Functions (kfuncs)](https://docs.kernel.org/bpf/kfuncs.html)
- [Linux 内核文档：BPF Iterators](https://docs.kernel.org/bpf/bpf_iterators.html)
- [Linux 内核文档：XDP RX Metadata](https://docs.kernel.org/networking/xdp-rx-metadata.html)
- [Linux 内核文档：Extensible Scheduler Class](https://docs.kernel.org/scheduler/sched-ext.html)
- [Linux 内核文档：BPF Design Q&A](https://docs.kernel.org/bpf/bpf_design_QA.html)
- [Eunomia 每日报告：eBPF 加载器能相信内核版本号吗？](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)
- [Eunomia 每日报告：内核升级后，同一个 eBPF 对象还能保持原来的语义吗？](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)
