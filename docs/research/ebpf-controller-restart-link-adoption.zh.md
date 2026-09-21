---
date: 2026-09-21
slug: ebpf-controller-restart-link-adoption
title: "eBPF 控制面重启后，怎么避免把数据路径挂两遍？"
description: "Pinned BPF link 可以比加载它的进程活得更久，但控制面重启后可能丢失足够的身份信息，无法安全接管已有对象。本文提出 restart receipt、kernel-first reconciliation 与对抗式重启测试。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - Lifecycle
  - libbpf
  - bpffs
research_question: "当用户态 eBPF 控制面进程重启，而内核、pinned link、map 与 attach target 都仍然存活时，系统应如何安全接管已有数据路径，避免重复 attach 或误接管错误的对象？"
source_cutoff: 2026-09-21
status: daily-report
---

# eBPF 控制面重启后，怎么避免把数据路径挂两遍？

设想一个网络守护进程加载了 eBPF 程序，通过 BPF link 把它挂到数据路径，并把 link pin 到 `bpffs`。随后用户态进程崩溃了。

内核此时完全按设计工作。Pinned link 继续持有 attachment 的引用，所以创建它的进程退出后，程序仍然可以继续运行。几秒后，新的守护进程启动并尝试恢复现场。

真正困难的问题从这里开始：**新进程怎么证明自己看到的 attachment 就是应该接管的那一个？**

`/sys/fs/bpf/foo/ingress` 这样的路径看起来像一个稳定答案，但它只是内核状态的一种视图。新进程可能处在不同的 mount namespace；同一路径可能已经解析到另一个 `bpffs` 实例；pin 可能已经消失，但 link 仍被其他引用保持；网卡或 cgroup 可能被删除后以同样的人类可读名称重新创建；legacy attachment 也可能与 link-based attachment 同时存在。更麻烦的是，两个控制面实例还可能同时尝试恢复。

Cilium 的一个实际问题把这个边界展示得很清楚。在 [Cilium issue #47847](https://github.com/cilium/cilium/issues/47847) 中，系统在 Cilium 运行期间又把一个新的 `bpffs` mount 覆盖到了 `/sys/fs/bpf`。Agent 重启后，新进程看不到旧 mount 视图里的 pin，于是创建了新的 TCX attachment。随后 `bpftool` 能在同一设备上看到重复的 ingress 与 egress TCX 程序。报告者最后确认 shadow mount 是稳定复现条件，并把问题作为环境行为而不是 Cilium bug 关闭。

这个结论反而很有价值。这里不需要 BPF link 的生命周期语义出错。只要两个本来都合理的机制对“对象身份”产生不同看法，就足够造成重复数据路径：

1. 内核仍然保留着旧的 live attachment；
2. 重启后的用户态控制面通过另一个路径视图判断旧 link 已经不存在。

因此，控制面恢复需要比“把 link pin 起来，然后重启后重新打开同一路径”更强的契约。一个真正可重启的 eBPF 控制面，在创建、更新或删除对象之前，应该先核对 **kernel-visible attachment state、pin namespace 身份、attach target 身份以及控制面 generation**。

本文讨论的边界很窄。它不同于[判断一个 eBPF object 能否在当前内核上被接受](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)，也不同于[内核升级后重新证明同一 object 语义是否仍然成立](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)。它也不是[有状态 eBPF 应用的事务式升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)，因为这里并不打算切换程序 generation。预期的数据路径可以完全不变，只有用户态控制面死掉并重启，而原来的内核对象图仍然存活。

<!-- more -->

## Pinning 解决的是生命周期，不是恢复身份

Linux 对 BPF object 使用引用计数。当前的 [eBPF syscall 文档](https://docs.kernel.org/userspace-api/ebpf/syscall.html)说明，`BPF_OBJ_PIN` 会给 BPF object 增加一个文件系统引用，因此原始文件描述符关闭后，对象也不会因为失去最后一个进程内引用而被回收；`BPF_OBJ_GET` 则可以重新打开 pinned object。相同 API 还提供 `BPF_LINK_GET_NEXT_ID`、`BPF_LINK_GET_FD_BY_ID` 和 `BPF_OBJ_GET_INFO_BY_FD`，用户态可以不依赖记忆中的 pin pathname，直接枚举并检查仍然存活的 link object。

对于 attachment，`BPF_LINK_CREATE` 返回管理 link 的文件描述符，`BPF_LINK_UPDATE` 可以在不拆掉再重建 attachment 的情况下替换关联程序。libbpf 的 [`bpf_link__pin()`](https://libbpf.readthedocs.io/en/latest/api.html) 文档也把进程退出后的行为说得很直接：pin 会增加 link 的引用，因此创建它的进程结束后，link 仍然可以留在内核中。

这是很强的基础机制，但它并没有自动变成完整的 restart protocol。

Pinning 回答的是：

> 原始文件描述符关闭之后，这个内核对象还要不要继续被引用？

而控制面恢复还要回答：

> 这个 live object 真的是我的吗？
>
> 它还挂在我认为的那个 target 上吗？
>
> 它属于我准备运行的 program generation 和 policy generation 吗？
>
> 当前 `/sys/fs/bpf` 看到的还是前一个控制面使用的那个 `bpffs` 吗？
>
> 如果 pin 不见了，究竟是 attachment 不存在，还是我从当前 namespace 看不到它？

这些并不是同一个问题。

## 生产级 loader 里其实已经存在一套隐含的恢复协议

当前 Cilium 的实现让这个边界非常具体。在 [`pkg/datapath/loader/tcx.go`](https://github.com/cilium/cilium/blob/main/pkg/datapath/loader/tcx.go) 中，TCX loader 会先按照预期的 pinned link 尝试 update。如果 link 存在，就直接更新 program；如果 pin 不存在或者 link 已经 defunct，才退回到创建新的 TCX link，并把新的 link pin 到约定路径。

代码还明确写出了关键生命周期语义：TCX link 成功 pin 后，用户态关闭 link handle 并不会把程序 detach。

只要 pathname 始终忠实地表示这个控制面管理的 live link，这个设计是合理的。Issue #47847 展示的正是这个假设失效时会发生什么。第二个 `bpffs` mount 能让路径看起来完全为空，同时旧 link 仍然在内核里 attach 着。此时“找不到 pin”实际表达的是“通过当前 mount 看不到”，并不能推出“系统里没有这个 managed attachment”。

这是一个常见的系统恢复模式：controller 通过一套 metadata plane 重建 ownership，而真正的 effect 存在另一套 plane 中。如果 metadata plane 丢失、换了 mount、变旧或只恢复了一部分，直接重新创建 effect 就可能产生重复副作用。

内核已经暴露了比 pathname 更强的观测能力。`BPF_LINK_GET_NEXT_ID` 与 `BPF_LINK_GET_FD_BY_ID` 可以枚举 live link，`BPF_OBJ_GET_INFO_BY_FD` 可以检查对象信息，`BPF_PROG_QUERY` 能对多种 attach target 查询关联 program。`bpftool link show` 这类工具本身就在利用 kernel-visible object state 做检查。

真正缺少的是一套通用规则：这些观测结果满足什么条件时，可以把一个旧对象**安全接管**为当前控制面所有。

## 同一个路径可以突然指向另一个 BPF 文件系统

Cilium 的复现很有代表性，因为它不依赖罕见的内核 race，而只是普通 Linux mount 语义。

Pathname 会根据调用进程的 mount namespace 解析。如果又把一个 `bpffs` instance mount 到 `/sys/fs/bpf`，之后这个位置的 path lookup 就会看到新的文件系统。之前的 mount 以及其中 pinned 的 object 可以仍然存活，甚至还通过其他引用保持可达，只是新控制面的 `/sys/fs/bpf/...` 已经解析到另一套对象。

因此下面这种启动逻辑不能被当成通用安全不变量：

```text
if expected_pin_exists():
    update_it()
else:
    attach_new_link()
```

“pin 不存在”的安全解释其实弱得多：

```text
在当前 pathname view 中看不到预期 pin
```

创建 replacement effect 之前，控制面还需要查询内核，判断相同 attachment 是否其实已经存在。

这和存储恢复很相似。目录项找不到，不代表某个外部 effect 从未发生。Recovery 必须把持久 effect 与 intent record 对起来。

## Link ID 是重要证据，但单独一个 ID 仍然不够

一个直觉上的改进是把 BPF link ID 记录下来，重启后使用 `BPF_LINK_GET_FD_BY_ID` 重新打开它。

这比只看路径可靠，但一个整数仍然不足以表达控制面想要的数据路径。

至少需要四层身份。

### 1. Attachment 身份

控制面要确认 hook 和 target，而不仅是某个 link object。具体字段依赖 program type，可能包括：

- attach type 与 hook 类型；
- network namespace 与 interface identity；
- cgroup identity；
- tracing target；
- TCX ordering 或其他 multi-program position；
- 该 hook 预期允许几个 managed attachment。

Interface name 与 cgroup path 都不一定是稳定身份。网卡可以被删除后用同一个名字重建，但 ifindex 已经不同；旧 cgroup 消失后，相同路径也可以指向一个新 cgroup。恢复契约应尽量使用该 hook 能提供的最强 kernel identity，并明确记录哪些 attachment 类型目前缺少足够的身份信息。

### 2. Program 身份

控制面还要确认 link 引用的到底是哪一个 program。可利用的证据包括 program ID、program tag、BTF 相关信息、预期 program name，以及控制面保留的 artifact hash。

Program name 只是短标签，并不是安全身份。Kernel object ID 也只是当前内核生命周期里的标识。控制面必须把 live object 重新连接到创建它的 artifact 与 configuration generation。

### 3. State 身份

真实数据路径通常不止一个 program。它可能依赖 pinned map、map-in-map、configuration map、policy generation、tail-call target，或者与另一个 program generation 共享状态。

如果控制面接管了正确的 link，却把它错误地配到另一个 map generation，后果甚至可能比重复 attach 更难发现。Program 会继续运行，只是读取了不同 ownership 或不同语义的 state。

### 4. Pin namespace 身份

恢复系统需要记录足够的信息，用来判断当前 `/sys/fs/bpf` 是否仍然是旧控制面使用的同一个文件系统 instance 或 mount view。具体编码可以由实现决定，但 pathname 本身不够。

政策上的要求很简单：**pin namespace 一旦发生变化，pathname-only 的“不存在”结论就必须失效。**

## 重启 race 会把恢复问题变成分布式 ownership 问题

Daemon restart 在架构图里经常看起来是单进程操作，生产环境里却可能有并发。

Kubernetes 可能在旧 helper process 完全退出前启动 replacement pod；supervisor 可能快速重试；operator 可能同时运行诊断或修复进程；host agent 也可能有另一个组件管理部分 link。这样一来，两个 controller generation 就可能有重叠时间窗口。

如果两边都执行：

```text
observe no expected pin
create link
pin link
```

即使 `bpffs` mount 从未变化，只要 observation 和 creation 没有被串行化，就仍然可能产生重复 effect。

所以更深一层的契约并不只是 object persistence，而是**reconciliation 的 single-writer ownership**。

重启后的控制面需要某种 generation 或 lease，使它在修改数据路径前能够证明以下条件之一：

- 自己是唯一被允许 reconcile 这组 attachment 的 controller generation；
- 或者，本次 mutation 对刚刚观测到的精确 kernel state 是幂等的。

对于很多 hook，第二个条件并不好实现，因为再 attach 一个有效程序本来就是合法的内核操作。内核无法知道这到底是管理员有意组合多个程序，还是旧控制面尚未消失时新进程意外挂了第二份。

## 真实 incident 也说明“进程健康”不等于“恢复完成”

另一个 Cilium 社区问题 [issue #46065](https://github.com/cilium/cilium/issues/46065) 描述了 `cgroup_inet_sock_release` link 在 agent restart 后仍然可见，并在新的 agent 进入 retry loop 时出现累积。这个 issue 最终没有形成一个持续维护的修复，因此不能把它当成已经确认的 kernel defect。它仍然是一个有用的 production observation：用户态进程可以看起来正常运行，甚至高层 status 也可能显示 broadly healthy，而 datapath reconciliation 实际仍处于失败或不明确状态。

这给出一个容易遗漏的 readiness 规则：

> 重启后的 eBPF control plane 不应该因为能重新 load program 就宣告 ready。只有预期的 live attachment graph 已经完成 reconcile，datapath 才真正恢复完成。

对一个网络 agent 来说，acceptance condition 可以包括：

- 每个 managed hook 上恰好存在预期的 effective attachment；
- 没有意外的旧 managed generation 继续执行；
- 所有必要 map 都已打开并通过 schema 检查；
- program-to-map 与 link-to-program 关系符合当前 generation；
- 不存在 ownership 无法分类的 live attachment。

Unknown state 应当明确保留为 unknown。把它变成“再 attach 一份看看流量能不能走”会破坏下一次故障分析最需要的证据。

## 现有研究还缺什么

Linux 已经给出了对象层 primitive，生产 loader 也有各自的恢复逻辑。真正薄弱的是两者之间的 adoption layer。

### Pin path 是名字，不是 ownership proof

Pin 让另一个进程重新拿到 object reference，但它本身无法证明这个 object 属于哪个 controller generation，也不能证明当前 pathname view 与旧进程看到的是同一个 mount。

### Kernel enumeration 只能观测，不能表达应用不变量

内核可以列出 link、program、map 以及对应信息，但它不知道应用级不变量，例如“这个 controller generation 对 eth0 ingress 必须恰好有一个 `cil_from_netdev` attachment，而且必须使用这一组 map”。

这个约束只能由 userspace 定义。

### 不同 hook 暴露的恢复接口并不一致

Modern BPF link 比旧 attachment mechanism 更容易枚举与管理，但真实应用往往混用 link、legacy TC attachment、cgroup attach API、基于 perf event 的 tracing、XDP，以及其他 subsystem-specific attachment model。

一个通用恢复层需要 hook-specific adapter，同时保持一致的 application-level ownership model。

### Reconciliation 很少被当作 state-space 问题测试

大多数 integration test 会验证 clean start、clean stop，最多再加一次 restart。真正困难的 failure state 通常出现在中间：

- attach 成功后、pin 之前进程死亡；
- pin 成功后、controller metadata commit 之前进程死亡；
- 重启前 mount namespace 发生变化；
- 只剩部分 link 可见；
- target 用相同人类可读名称重新创建；
- 两个 controller 同时 recovery；
- link 指向旧 program generation，而 map 已经属于新 generation；
- legacy 与 link-based attachment 同时存在；
- controller 已拿到新 reference 后 update 又失败。

Restart contract 应该在这类状态空间里接受测试，而不是只验证一次干净重启。

## 兼具学术价值与生产价值的方向

### 1. 为 live BPF object graph 建立 restart receipt

第一个方向是控制面在 datapath generation 激活之后写出一份 **restart receipt**。

它不需要 dump 每一个 BPF object，而应记录未来判断“接管、更新、隔离还是重建”所需的最小 ownership graph。

例如每个 managed attachment 可以保留：

```text
controller_generation: 184
artifact_digest: sha256:...
policy_generation: 9271

pin_namespace:
  expected_bpffs_mount_identity: ...
  expected_pin: /sys/fs/bpf/app/eth0/ingress

attachment:
  kind: tcx/ingress
  target_identity: netns + ifindex + stable device evidence
  expected_multiplicity: 1
  expected_order: ...

link:
  observed_link_id: 314
  program_id: 9256
  link_info_digest: ...

program:
  program_tag: ...
  btf_id: ...
  artifact_section: ...

state:
  map_manifest_digest: ...
```

Receipt 应该存放在 `bpffs` 之外，同时引用 `bpffs` 中的对象。如果 receipt 和 pin 都只存在于同一个被 shadow 的 BPF filesystem 中，mount 变化会同时隐藏 object 与解释 object 的唯一记录。

学术问题是：这份 receipt 最小能做到什么程度，同时仍足以避免 false adoption？不同 hook 能提供的 target metadata 不一样，因此原型可以设计 common core，再配合 hook-specific identity field。

生产价值则很直接：control plane restart 可以解释“为什么接管了这个 link”或“为什么拒绝自动修复”，而不是只凭路径是否存在做猜测。

#### 如何评测

建立一个覆盖 TCX、cgroup link、XDP 或另一个 network hook，再加一种 tracing link 的测试矩阵。每次 restart 比较四种策略：

1. 只看 pin path；
2. 只记录 link ID；
3. kernel enumeration 加 program/link name；
4. 同时校验 target、program、state 与 mount identity 的 receipt-based recovery。

注入 target recreation、mount namespace 变化、部分 pin 删除和 stale controller metadata。测量 false adoption、不必要的重新 attach、无法消除的 ambiguity，以及 recovery latency。

如果 receipt 仍然无法区分常见 stale-state case，或者最终膨胀成内核状态的完整复制，这个方向就失败了。

### 2. 把启动流程改成 kernel-first reconciliation transaction

第二个方向改变 control plane 的操作顺序。

Restart 不应该从“预期 pin directory 应该是完整的”这一假设开始，而应该先读取 kernel ground truth。

一种协议可以是：

```text
1. 获取 controller-generation ownership
2. 检查 bpffs mount identity
3. 从内核枚举或查询相关 live attachment
4. 在任何 mutation 之前先打开 candidate link/program/map 的 FD
5. 把 candidate 与 restart receipt 匹配
6. 对每个 expected attachment 分类：
      完全匹配 -> adopt
      program 旧但 link 可安全更新 -> update existing link
      有充分证据证明不存在 -> create
      ambiguous / duplicate -> quarantine 或 fail closed
7. 在当前 intended bpffs 中为 adopted object 修复或重新建立 pin
8. 验证完整 effective attachment graph
9. 标记 datapath ready
10. 只回收已经证明 stale 的对象
```

第 6 步是整个协议的安全核心：**“找不到 pin”不能单独成为“创建新 attachment”的证据。**

如果 hook 支持 `BPF_LINK_UPDATE`，接管已有 link 后可以原地更新 program，避免先拆再挂造成 continuity gap。对于存在 query API 的 hook，控制面即使看不到原来的 pin，也能先识别已经执行中的 kernel effect。

协议还把 ambiguity 变成显式状态。如果两个 live link 都很像预期的那个 attachment，而控制面无法证明哪一个连接着当前 state，自动删除其中一个并不安全。正确的系统可以 fail closed、把节点标为 degraded，或者进入 hook-specific repair policy，而不是随机猜一个。

#### 如何评测

在协议的每个 transition 后都杀掉 controller，然后重启。验证的不变量应该比“最终恢复健康”更强：

- 除非 hook contract 明确允许多个，否则预期 effective attachment 最多只有一个；
- 不会把 policy generation 与不兼容 state 配在一起；
- 没有 active attachment 会在缺少 ownership proof 时被删除；
- 在有界 recovery interval 后，每个 managed object 都被分类成 adopted、replaced、stale 或 unknown。

如果主要生产 hook 的 kernel introspection 无法暴露足够 target identity，这个方案可能无法统一。这个负结果同样有价值，因为它会指出哪些 attachment 类型真正需要新的 kernel introspection。

### 3. 为 eBPF control plane 建立对抗式 restart benchmark

第三个方向不是增加新 API，而是建立评测 artifact。

Restart benchmark 应该像 crash-consistency test 一样测试 BPF lifecycle recovery。它不能只在干净位置重启 daemon，而要在每个 externally visible transition 周围注入失败。

值得覆盖的 perturbation 包括：

- link create 后、pin 之前发送 `SIGKILL`；
- pin 后、controller-state commit 之前发送 `SIGKILL`；
- restart 前覆盖或更换 `bpffs` mount；
- 在不同 mount namespace 中重启；
- 删除一个 pin，但通过其他 reference 保持 link 存活；
- 用同一名称重新创建 interface 或 cgroup；
- 同时启动两个 controller generation；
- 强制 `BPF_LINK_UPDATE` 失败；
- 保留旧 link，同时切换一个 map generation；
- 让 replacement process 拥有不同权限。

Benchmark 必须获取 kernel-side ground truth。每次 recovery 前后都枚举 link/program，并在可能时查询 effective attachment state，再真正驱动 packet、syscall 或 traced event 经过 hook，统计 program 到底执行了几次。

最有用的 metric 不只是 startup latency，还包括：

- duplicate effective attachment count；
- orphan lifetime；
- wrong-generation execution count；
- policy discontinuity duration；
- false adoption rate；
- 错误删除仍被拥有的 object 的次数；
- unknown state 持续时间；
- 无需 operator repair 就能收敛的 crash point 比例。

生产控制面最后应该能够明确给出自己的 restart envelope：哪些 crash point 与 namespace mutation 可以自动恢复，哪些能检测但拒绝自动 repair，哪些仍然不支持。

如果 benchmark 只数 object，却没有验证 packet、syscall 或 trace event 是否被实际处理了两次，那么它仍然测错了问题。Ground truth 必须覆盖 effect execution，而不仅是 bookkeeping。

## 今天就能采用的更严格恢复策略

不需要等待新的 kernel API，现有生产实现已经可以把 restart policy 做得更严格：

1. **如果 attachment 本来就应该跨进程存活，应优先 pin attachment object。** 只 pin map 或 program，无法给 control plane 同样直接的 attachment lifecycle handle。
2. **把 `bpffs` mount 当成带身份的配置，而不是一个字符串路径。** 在解释“pin 不存在”之前先检测 mount 是否发生变化。
3. **创建 replacement 前先查询 kernel attachment state。** 能用 link enumeration、link info 和 hook-specific query API 时，先确认 effect 是否已经存在。
4. **按证据接管，不按名字接管。** 同时匹配 target identity、program identity、state generation 与 controller generation。
5. **保持一个 reconciliation owner。** 不让两个 controller generation 独立判断是否需要 recreate effect。
6. **Reconciliation 完成前不要宣告 ready。** Process health 与 datapath ownership 是两个状态。
7. **Unknown object 必须可见。** 无法归属的 live link 应产生明确 diagnostic，而不是被静默忽略。
8. **已有正确 attachment 且 hook 支持时，优先原地 link update。** 不要为了换 program 先制造第二个 attachment。
9. **主动测试 mount 与 namespace failure。** Shadow `bpffs` 已经产生过可稳定复现的 duplicate TCX，不应该再只被当成理论边界。

这些规则会让 restart 在一个地方变得更“不方便”：控制面无法证明 ownership 时，可能选择拒绝自动 self-heal。但这仍然优于一种看似自愈、实际上通过再挂一份 policy program 来掩盖旧 effect 的系统。

## 更一般的结论：持久化会自然产生 adoption 问题

Pinning 经常被描述成“让 BPF state 在进程退出后继续存在”。这当然没错，但一旦对象可以跨进程存活，control plane 的 correctness problem 也随之变化。

没有 persistence 时，进程死亡会释放最后一个 reference，cleanup 是隐式完成的。有 persistence 之后，新进程面对的是一个包含既存 effect 的世界，而这些 effect 并不是它亲手创建的。它必须有协议来识别并接管这些 effect。

其他系统也有类似转换。持久资源会把 correctness 从 object lifetime 推向 recovery identity。数据库需要 transaction recovery，orchestrator 需要 resource ownership 与 generation，可重启的 eBPF controller 也需要比“一个 pin directory”更完整的恢复契约。

内核已经提供了很强的基础模块：reference-counted BPF object、pin、link ID、object-info query、link update 与 attachment query。缺少的是 control plane 层面的一句话，而且这句话要能被机器验证：

> **这些是 generation G 的 live kernel effect；这些是它们对应的 target 与 state；新 controller 已经证明它们匹配，所以不需要再创建第二份 effect。**

如果这个 contract 能被明确实现，eBPF lifecycle recovery 就不必继续依赖 pathname convention、hook-specific cleanup 和 best-effort startup reconciliation 的组合。

## 哪些结果会改变这个判断？

有三类结果会削弱本文对更强 adoption contract 的需求。

第一，如果已经使用 persistent BPF link 的生产 control plane 能在 realistic restart、namespace mutation 与 target recreation 测试里证明，只依赖稳定 mount 配置和 pin-path reopening 就足以消除 duplicate 与 orphan attachment，那么复杂 receipt 的收益可能不足以覆盖成本。

第二，如果主要 hook 上的 kernel attachment enumeration 无法做到足够完整、便宜且稳定，那么 kernel-first reconciliation 可能只能维持 hook-specific implementation，而不适合抽象成统一 control-plane layer。

第三，如果未来出现一个更简单的 kernel primitive，可以原子地把 persistent link 与 application-defined owner generation 绑定，并能通过 query API 读回这份 ownership，那么大量 userspace receipt machinery 都可以被压缩掉。

在这些证据出现之前，更符合当前机制与 incident 的判断仍然是：`bpffs` pinning 是 lifetime mechanism，而 restartable eBPF control plane 还需要一个 **adoption protocol**。它必须先 reconcile live kernel object graph，再决定是否真的需要创建另一个对象。

## 参考资料

- [Linux kernel documentation: eBPF syscall commands](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [libbpf API documentation](https://libbpf.readthedocs.io/en/latest/api.html)
- [Cilium TCX loader implementation](https://github.com/cilium/cilium/blob/main/pkg/datapath/loader/tcx.go)
- [Cilium issue #47847: shadow bpffs mount and duplicate TCX links after restart](https://github.com/cilium/cilium/issues/47847)
- [Cilium issue #46065: reported orphaned cgroup BPF link behavior across agent restarts](https://github.com/cilium/cilium/issues/46065)
