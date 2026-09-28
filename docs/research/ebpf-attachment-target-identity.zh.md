---
date: 2026-09-28
slug: ebpf-attachment-target-identity
title: "eBPF 挂载能自动跟随被重建的工作负载吗？"
description: "BPF link 绑定的是具体内核目标，而容器、cgroup、namespace 和网卡可能在同一个逻辑工作负载下被重建。本文讨论如何证明跨目标代际的挂载连续性。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - BPF link
  - Kubernetes
  - Compatibility
research_question: "当逻辑工作负载仍然存在、但内核挂载目标被重建时，eBPF 控制器怎样证明观测或网络处理确实跟随到了新目标，而且没有静默覆盖空窗？"
source_cutoff: 2026-09-28
status: daily-report
---

# eBPF 挂载能自动跟随被重建的工作负载吗？

假设一个 agent 把 eBPF 程序挂到某个容器的 cgroup 上，并把得到的 BPF link pin 到 bpffs。loader 退出以后 link 仍然存在，于是从运维视角看，"控制器重启后挂载仍然活着"。

随后 Kubernetes 重建了这个 Pod。

新的 Pod 可以承担完全相同的业务角色，甚至保留相同的名字，但 Kubernetes 会给它新的 Pod UID，而且它还可能被调度到另一台机器。在 Linux 层，新实例对应的是新的 namespace、cgroup、网络设备以及其他内核对象。旧 BPF link 并没有保存一个 Kubernetes selector，让内核自动理解“请跟随这个工作负载的下一代实例”；它绑定的是一个具体内核目标。

因此这里有一个容易被忽略的生命周期边界：**BPF 对象还活着，不等于逻辑挂载关系仍然连续。**

内核 API 本身已经把这个区别暴露出来。`BPF_LINK_CREATE` 把程序挂到某个 target，并返回 link fd。cgroup 程序的目标是某个具体 cgroup 目录的 fd；XDP 使用网络接口的 index；其他 link type 也各自引用具体目标。pin 一个 link 可以让它跨越 loader 进程的生命周期，但 pin 并不会把这个目标变成“逻辑工作负载 selector”。

本文继续 [eBPF Deployment Compatibility and Lifecycle](https://eunomia.dev/zh/research/ebpf-kernel-interface-negotiation/) 系列。前面的报告讨论过主机能力准入、内核升级后的语义兼容，以及面对变化中的 kfunc/iterator/`struct_ops` 接口时如何选择程序变体。这里即使程序本身完全兼容，仍然会出问题：**程序原来挂载的那个对象已经被替换了。**

<!-- more -->

## BPF link 解决了所有权，但没有解决逻辑目标自动重绑定

BPF link 相比旧式 attach API 的一个重要改进，是把一次挂载本身变成有明确 fd 生命周期的内核对象。用户空间 API 把 `BPF_LINK_CREATE` 定义为：将程序挂到 `target_fd` 所代表的目标上，并返回一个用于管理该 link 的 fd。和其他 BPF 对象一样，link 还可以 pin 到 bpffs，从而不依赖原 loader 进程继续存活。

这正好解决了长时间运行的 observability 或 networking agent 在“控制器进程重启”场景下的对象所有权问题。

但 API 也同时给出了边界：

```text
逻辑意图
    "观测 checkout-api 工作负载"

        解析

内核目标代 G17
    cgroup fd / netns fd / ifindex / 其他目标 handle

        attach

BPF link L42 -> 目标代 G17
```

如果工作负载被替换，同一个逻辑意图可能重新解析到 G18：

```text
逻辑意图
    "观测 checkout-api 工作负载"

        再次解析

内核目标代 G18

        attach again

BPF link L43 -> 目标代 G18
```

让 `L42` 一直存在，不能证明 `G18` 已经被覆盖。

这也不是 Kubernetes 特有的问题。Linux 必须知道程序到底运行在哪个真实 hook 上，所以 attach API 天然接收具体目标。libbpf 的 cgroup attach 接收 `cgroup_fd`；TCX 与 XDP 通过 `ifindex` 指定网卡；涉及 network namespace 的 hook 会使用 namespace fd。这些标识很有用，正是因为它们指向真实内核对象，而不是抽象的服务名。

## target lifetime 本来就是 BPF 语义的一部分

当前 Linux 的 cgroup/BPF 实现直接把目标生命周期当作内核问题处理。`kernel/bpf/cgroup.c` 注册了 cgroup lifetime notifier，处理 cgroup online/offline 事件，同时还为 cgroup BPF 状态准备了独立的销毁工作队列。

这里不应该得出“所有 BPF link 的销毁行为都一样”的结论；不同 link type 的生命周期规则并不相同。更准确的结论是：**挂载的生命周期会受其 target type 生命周期规则约束。**

这会带来两个常见误判。

第一，控制器不能仅凭一个 pinned link 仍然存在，就判断目标工作负载仍在被观测。pin 能证明某个内核 BPF 对象还有引用，却不能证明“当前应该被观测的工作负载”仍然就是原来的 target。

第二，也不能把一个看起来可重复使用的 kernel identifier 当作持久工作负载身份。`ifindex` 是网络对象标识，不是 Pod UID；cgroup path 是层级中的位置，也不能单独证明这个位置背后的进程集合仍然属于最初那一代实例。

因此更准确的问题应该是：

> 哪种 identity 表示长期逻辑意图，哪种 identity 表示当前内核目标，以及两者之间的映射要用什么证据来证明？

## Kubernetes 把这三个身份层次暴露得很明显

Kubernetes 官方文档明确把 Pod 描述为相对短暂、可替换的对象。替换后的 Pod 可以和旧 Pod 同名，却拥有不同 UID，并且可能落到另一台 node。Pod 的运行上下文本身又包含 Linux namespace、cgroup 等资源。

对于 eBPF 系统，可以把 identity 至少分成三层：

1. **逻辑工作负载身份。** Deployment、DaemonSet、StatefulSet 成员、service role、tenant、selector，或者别的运维意图。
2. **编排器代际。** 某个具体 Pod UID、sandbox、container attempt 或 rollout generation。
3. **内核目标身份。** 实际被 BPF 程序 attach 的 cgroup、namespace、network interface、task、socket 或其他对象。

生产控制器通常三层都需要。如果只保存第 3 层，就难以回答这个内核对象是否仍属于目标工作负载；如果只保存第 1 层，就无法证明程序到底挂到了哪里；如果只保存第 1 和第 3 层，却没有记录代际切换，就可能漏掉 replacement window。

即使只是 observability，这个区别也很重要。短暂缺失一些样本可能是可接受的，但应该被明确测量成一个 coverage gap，而不是被描述成“持续覆盖”。

## “挂到更高层再过滤”是很强的 baseline，但不是万能答案

一个简单而且经常很好的替代方案，是尽量不要挂到短命对象。可以把程序挂到生命周期更长的祖先 cgroup 或 node-global hook，再在 BPF 程序里根据 cgroup ID、namespace、mark、address 或 map state 对事件分类。

这样做能显著减少 attachment churn，也可能让 target replacement 对 hook 本身几乎不可见。

但连续性问题并没有消失，只是移动到了分类状态：

```text
node-global hook
    -> 收到事件
    -> 把 kernel identity 解析成逻辑工作负载
    -> 查当前 workload generation
    -> 记录 / 处理
```

此时危险窗口从“新 link 还没挂上”变成“新 target identity 还没有进入映射”，或者“复用的 identity 仍然指向旧 generation”。

它还有不同的成本和隔离取舍。更宽的 hook 会看到大量无关 workload 的事件，增加 map 与分类压力，也扩大错误配置的影响范围。有些 BPF program type 本来就是 target-specific，也不能简单提升成一个全局 hook。

因此研究问题不应是“所有 agent 都应该局部 attach 还是全局 attach”。更有价值的问题是：**局部 attachment 和 broad-hook filtering 能否共享一套显式 continuity contract，而不是各自维护隐式的重建逻辑？**

## 现有研究还缺什么

### link 还活着，并不能证明当前工作负载被覆盖

BPF object discovery 可以告诉我们 link 是否存在，也能暴露一些 kernel-side link 信息；orchestrator 可以告诉我们现在应该存在什么 Pod 或 container generation。缺少的是一个标准化证据，把这两句话连接起来。

例如可以记录：

```text
intent: workload selector / generation
orchestrator: pod UID / container attempt
target: kind + kernel identity + node
link: link ID + program identity
resolved_at: monotonic generation
state: prepared | active | retiring | stale
```

一个有区分力的测试，是持续快速替换工作负载，同时从独立路径检查内核 target 和实际 BPF observation。如果 controller 报告“covered”，但当前 target 并不存在匹配的 active attachment 或 classification entry，那么这套 evidence model 就不够强。

### replacement 是一次 transition，但 attach API 通常只暴露两个端点

内核 attach API 操作具体 target 是正确的，Kubernetes controller reconcile desired state 也是正确的。缺少的是 old generation 与 new generation 之间的 transition protocol。

对于监控场景，“旧目标消失 -> 发现新目标 -> attach”可能已经足够；但如果要求更强的连续性，这中间就是可观测空窗。更好的顺序可能是先准备新目标，再退休旧目标，但前提是 agent 能在应用把新 generation 当作 ready 之前完成发现和初始化。

所以缺少的并不是另一个 attach syscall，而是 workload lifecycle 与 BPF attachment readiness 之间的协调边界。

实验不应该只测平均 reconcile latency，而应在 target discovery、BPF load、link create、map init、orchestrator watch delivery、old-target retirement 各阶段注入 delay 和 failure，分别测 uncovered time 与错误 overlap。

### kernel identifier 与逻辑 identity 的复用规则不同

逻辑名字本来就允许复用；kernel identifier 在旧对象销毁后也可能被回收。只缓存 `ifindex`、cgroup path、PID 或某个局部数字，很容易在高 churn 下把新对象误认成旧对象，除非把这个标识和更强的 generation context 绑定起来。

这是一个更一般的系统问题：一个 identifier 只有放回“它在哪段 lifetime 内保证唯一”这个条件里才有意义。

合适的 benchmark 应主动制造快速 create/destroy/recreate，并尽量触发 identifier reuse。正确系统应该证明：旧 generation 的状态不会仅仅因为一个数字或 pathname 再次出现，就自动对新 workload 生效。

### 缺少统一的 attachment continuity correctness metric

现在很多系统能报告 load success、attach success 或 controller reconcile time。这些都是运维指标，却没有直接回答“哪一代 workload 的哪些操作，实际被哪一代 BPF attachment 覆盖”。

没有一条 ground-truth workload/target timeline 时，两个系统都可以宣称 reconcile 成功，却有完全不同的 blind window。

因此还缺一个把 target replacement 当作一等 fault 的 benchmark 与 trace schema。

## 兼具学术价值与生产价值的方向

### 方向一：为每次逻辑挂载生成 target-generation receipt

**Gap。** Controller 知道工作负载意图，kernel attach API 知道具体 target，但两者映射通常是隐式的，replacement 之后难以审计。

**Mechanism。** 每次 resolve attachment 时都生成一个 target-generation receipt，绑定：

- 稳定的 logical selector 与应用 generation；
- Pod UID、container attempt 等 orchestrator identity；
- target type 与 node；
- target-specific kernel identity；
- program/link identity 与创建结果；
- 如有的话，前一个 generation。

target-specific identity 应尽量使用能表达 lifetime 的 tuple，而不是一个裸数字。例如网络设备可以把 interface identity 和 network namespace、观测到的 generation 绑定，而不是把 `ifindex` 当成全局永久标识；cgroup 可以保留已经打开的 target，并另外记录 kernel-visible cgroup identity 和 orchestrator ownership。

Receipt 是证据，不是新的 kernel primitive；真正 attach 时 target fd 和内核仍然是最终权威。

**与已有工作的差异。** BPF link 显式化 attachment object lifetime，Kubernetes UID 显式化 orchestrator object lifetime。这里的机制把两个 lifetime domain 连接起来，并保存 generation transition。

**Artifact。** 一个小型 libbpf-side identity library、cgroup/netns/netdevice target resolver，以及可被 `bpftool` 风格命令查看的机器可读 receipt。

**Evaluation。** 高频重建 Pod、container、cgroup、namespace、veth，对比 path-only、numeric-ID-only、link-only 与 generation-bound tracking。测 false continuity、stale-object match、诊断时间、receipt size 与 resolver overhead。

**Academic value。** 一般化问题是：如何组合多个只在各自 lifetime domain 内保证唯一的 identity。

**Production value。** rollout 或事故之后，operator 能回答“这个 link 实际覆盖的是哪一代 workload？”

**Failure condition。** 如果现有 link metadata 加 orchestrator state 已经能在高 churn 下便宜且可靠地重建映射，专门的 receipt format 就没有足够价值。

### 方向二：面向 replacement 的双代 reconcile protocol

**Gap。** 普通 controller 强调 eventual reconciliation，但更强的 continuity 需要定义 old/new target generation 如何交接。

**Mechanism。** 把 replacement 建模成两代 transition：

```text
resolve G(next)
    -> 为 G(next) 创建并初始化 attachment
    -> 验证 target + program + workload generation
    -> 标记 G(next) attachment-ready
    -> 允许正常 workload traffic / measurement
    -> retire G(prev)
```

在 hook 允许 overlap 时，old generation 可以一直保持 active，直到 next generation 准备好。如果某类 hook 无法 overlap，controller 至少应该显式记录 uncovered interval，而不是把它隐藏。

在 Kubernetes 中，instrumentation agent 可以暴露 readiness signal；对于真正要求连续性的 workload，平台可以等 required attachment receipt active 后再把新 generation 视为正常 ready。具体集成方式可以因 workload 而异，重要的是把 workload readiness 与 attachment readiness 放进同一个 transition model。

**与已有工作的差异。** 这不是八月那篇 transactional eBPF program upgrade：那里是逻辑 target 不变、program/state generation 变化；这里 program 完全可以不变，变化的是**内核 target 本身**。

**Artifact。** 带持久化 state machine 的 reconcile library、Kubernetes integration、fault injection，以及 cgroup/network-device target adapter。

**Evaluation。** 在 rollout、container restart、node drain、CNI recreation 和 controller restart 下，对比 naive watch-and-attach、periodic polling、broad-hook filtering 和 two-generation reconciliation。测 uncovered operation、duplicate observation、replacement-to-ready latency、API load，以及 lost watch event 后的恢复。

**Academic value。** 研究 kernel API 和 orchestrator 暴露不同 transaction boundary 时，resource replacement 如何保持可验证连续性。

**Production value。** Observability 与 networking agent 能把“最终会重新挂上”变成一个可测量 SLO。

**Failure condition。** 如果 broad stable hook 加 classification state 对几乎所有相关 program type 都能用更低复杂度消除 attachment gap，那么 target-specific transactional reconciliation 只适合少数场景。

### 方向三：为 ephemeral target 建立 coverage-witness benchmark

**Gap。** Attach success 与 controller latency 看不出 target churn 期间到底漏掉了哪些 workload operation。

**Mechanism。** Ground-truth harness 为每个 workload generation 和测试 operation 分配单调 identity，同时记录 target-generation receipt 与 BPF observation。Checker 对每个 operation 检查：

```text
expected logical workload generation
expected target generation
observed BPF attachment/classification generation
sample or result produced
```

Fault injection 可以销毁并重建 cgroup、network namespace、veth、Pod 和 node，延迟 orchestrator event，重启 controller，并制造 identifier reuse 压力。核心结果不再是“40 ms 内重新 attach”，而是哪一些 operation 没覆盖、重复覆盖，或者被归到了错误 generation。

**与已有工作的差异。** Linux BPF selftests 主要验证 kernel attachment API 自身是否正确；这个 benchmark 验证 logical orchestration identity 与 kernel target identity 之间的端到端 continuity。

**Artifact。** 可复现的 Kubernetes/Linux testbed、event trace schema、checker，以及 replacement fault corpus。

**Evaluation。** 在同样 churn 下运行多个真实 agent 或 prototype instrumentation path。报告 uncovered-operation rate、最大 blind interval、stale-generation rate、false attribution、controller CPU/API cost，以及 readiness coordination 带来的 workload latency。还应加入 observability-only workload，证明允许 bounded loss 的简单方案在合适场景里确实应该赢。

**Academic value。** 为“target 本身会变化”的 dynamic instrumentation system 提供 correctness metric 与 workload。

**Production value。** Operator 可以验证所谓 persistent attachment 是否真的经受 workload replacement，而不只是经受 controller restart。

**Failure condition。** 如果 realistic churn 下 uncovered operation 极少、也没有可测后果，那么更复杂的 continuity machinery 可能不值得生产成本。

## 生产控制器现在就应该区分哪些状态

即使完全不增加 kernel API，也至少应把这些状态分开：

```text
program loaded
link object alive
old target alive
logical workload desired
current target resolved
current target attached
current workload generation active
```

它们不是同义词。

Controller restart 后，通过 pinned link 重新发现并接管已有 attachment，属于 ownership recovery。

Workload replacement 后，判断同一逻辑意图是否应该跟随到新的 kernel target，属于 target reconciliation。

Host reboot 后，原来的 kernel object graph 本身就不存在了，属于 durability 与 reconstruction。

把三件事混在一起，会让系统看起来比实际更“persistent”。

## 哪些结果会改变这个判断？

最强的反例，是生产数据表明绝大部分 eBPF agent 都能挂在稳定的祖先或 node-global hook 上，而且 target replacement 只需要更新普通 classification state，不存在独立 continuity failure。如果 broad hook 在重要 observability/networking 场景里成本可控、identity 也足够明确，那么通用 target-rebinding protocol 就没有必要。

如果现有 link type 已经暴露足够强的 target-generation metadata，让 controller 无需额外 receipt 就能可靠发现所有重要 replacement，这个结论也会变弱。类似地，如果 orchestrator 本身保证 workload 在 node-local instrumentation 完成前绝不会进入正常 active 状态，也会降低额外协议的价值。

最后，严格 continuity 并不总是正确目标。Profiler 完全可能愿意接受很短的 blind interval，而不是为了消除它去延迟 workload readiness。机制应该把 gap 和成本暴露出来，而不是强迫所有系统采用同一种策略。

较窄但仍然有用的结论是：**persistent BPF link 证明的是 attachment object 的持久存在，不是逻辑 workload 到 kernel target 映射的持久存在。** 如果系统声称 workload replacement 期间仍能保持 observability 或 networking continuity，就需要为这个映射和 target generation transition 提供可检查的证据。

## Sources

- [Linux kernel documentation: eBPF syscall API](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [Linux kernel source: cgroup BPF lifecycle handling](https://github.com/torvalds/linux/blob/master/kernel/bpf/cgroup.c)
- [Libbpf API: attach a program to a cgroup](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_program__attach_cgroup/)
- [Libbpf API: TCX attachment by interface index](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_program__attach_tcx/)
- [Libbpf API: pin a BPF link](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_link__pin/)
- [Kubernetes documentation: Pod lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Kubernetes documentation: Pods and their Linux isolation context](https://kubernetes.io/docs/concepts/workloads/pods/)
- [Eunomia Daily Report: eBPF 加载器能把 kfunc 简化成“有”或“没有”吗？](https://eunomia.dev/zh/research/ebpf-kernel-interface-negotiation/)
- [Eunomia Daily Report: eBPF 加载器能相信内核版本号吗？](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)
