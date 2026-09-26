---
date: 2026-09-23
slug: ebpf-pinned-map-reboot-state
title: "Pinned eBPF Map 真的能跨主机重启存活吗？"
description: "bpffs pin 能让 eBPF map 跨进程重启继续存在，却不能跨主机重启保存状态。本文讨论状态契约、一致性 checkpoint 与恢复准入。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - BPF Maps
  - State Recovery
  - BTF
research_question: "当 bpffs pin 只保存对当前内核对象的引用时，eBPF 应用需要什么状态契约，才能在主机重启后安全恢复或重建 map 状态？"
source_cutoff: 2026-09-23
status: daily-report
---

# Pinned eBPF Map 真的能跨主机重启存活吗？

一个 daemon 创建 eBPF map，写入策略状态，把它 pin 到 `/sys/fs/bpf`，然后退出。新的 daemon 可以重新打开同一个 map 继续工作。看起来，这份状态已经“持久化”了。

主机一重启，这个判断就不成立了。

旧 map 已经随旧内核消失。系统重新 mount bpffs 后，路径空间可以回来，但原来的 BPF 对象和 map 内容不会因此重建。[eBPF pinning 文档](https://docs.ebpf.io/linux/concepts/pinning/)把边界写得很清楚：pin 可以让对象跨进程生命周期继续存在，但 system restart 之后并不会保留。

所以真正的问题不是“怎样重新打开一个 pin”，而是：哪些 map 状态必须 checkpoint，哪些应该从 controller database、配置或路由状态重建，哪些应该直接清空，以及怎样证明恢复后的状态在新内核、新拓扑和新应用版本下仍然有效。

本文主张给有状态 eBPF 应用增加一个明确的**重启状态契约**。它需要把 kernel-visible 的 map 兼容性和应用语义分开，给 checkpoint 定义一致性边界，并把恢复过程做成一次有验证的 staged promotion，而不是把旧字节重新塞回 map 就算成功。

这个问题与之前的[有状态 eBPF 原子升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)不同。事务式升级还有旧的 live object graph 可以参与 prepare、migrate 与 cutover；主机重启后，旧内核对象已经不存在。它也不同于[跨内核语义兼容](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)：后者验证同一 artifact 在新内核上是否保持行为，这里首先要证明重建出来的**输入状态**本身是可信的。

<!-- more -->

## Pin 保存的是对象生命周期，不是存储生命周期

BPF object 使用引用计数。`BPF_OBJ_PIN` 在 bpffs 中增加一个文件系统引用，所以创建对象的进程关掉 fd 以后，program、map、link 或 BTF object 仍然可以留在内核里。另一个进程随后可以通过 `BPF_OBJ_GET` 取得同一个 live object 的 fd。

这正适合 controller restart 和跨进程共享。它表达的生命周期大致是：

```text
用户态进程生命周期
    < pinned BPF object 生命周期
    <= 当前 kernel instance 生命周期
```

主机重启跨过了最后一条边界。重新 mount bpffs 只是重新建立用来命名 BPF 对象的文件系统，并不会把旧对象从磁盘恢复出来。

生产系统里，这个区别很重要。有些 map 只是 cache，重启后清空影响不大；有些 map 保存授权、policy generation、accounting 或恢复 cursor，清空以后可能直接造成 outage，甚至让 fallback 行为变得不安全。一个简单的 `persistent=true` 无法表达这些差异。

## 结构兼容不等于状态语义兼容

对于还活着的 pinned map，loader 可以检查现有对象是否和新的 BPF object 兼容，再复用它。libbpf 提供 `bpf_map__reuse_fd()` 一类 API，内核也能通过 `BPF_OBJ_GET_INFO_BY_FD` 暴露 map 元数据。[Linux BTF 文档](https://docs.kernel.org/bpf/btf.html)还允许 map 关联 key/value 的 BTF type。

这些机制可以回答：“新的 loader 能不能安全绑定这个 live kernel map？”

它们回答不了：“map 里的值现在还代表应用以为的意思吗？”

例如下面两个 value 都是 8 字节：

```c
struct policy_state_v1 {
    __u32 verdict;
    __u32 generation;
};

struct policy_state_v2 {
    __u32 verdict;
    __u32 lease_seconds;
};
```

只看 size 会认为完全兼容。BTF 能告诉工具 field 的结构，但应用升级也可能在不改变布局的情况下改变单位、有效期、ownership 或 field 的实际含义。而且 raw BTF type ID 只在自己的 BTF object 内有意义，也不适合作为长期 schema identity。

因此重启恢复至少需要两层证据：

```text
内核可见兼容性
  map type + size + flags + target support + BTF 结构

应用状态兼容性
  semantic schema + validity epoch + ownership + migration/rebuild rule
```

目标内核和 verifier 仍然决定重新创建的 BPF 对象是否合法；第二层必须由应用定义，因为内核没有办法判断 `generation = 7` 与 `lease_seconds = 7` 是否可以互换。

## 把所有 key dump 出来，也不一定得到一个 checkpoint

普通 hash map 也有这个问题。Linux 明确允许不同 CPU 并发访问 map value。用户态可以遍历 map，也可以用 batch API 提高读取效率，但这些接口并没有提供跨多个 key、多个 map 与 controller-side state 的事务快照。

假设一个策略应用保存：

```text
map A: active policy generation = 42
map B: generation 42 允许的 principals
```

checkpoint 程序先读 A。随后 controller 把两张 map 都更新到 generation 43。checkpoint 再去读 B。每次单独读取都成功，最终文件却组合出一个从未真实存在过的状态。

`bpf_spin_lock` 只能保护单个 value，解决不了跨多个对象的一致性。batch lookup 只是更快，也不等于 transaction。

因此更有意义的问题是：

> 恢复后的 state set 需要满足哪些 invariant？checkpoint 用什么协议证明这些数据属于同一个 recovery cut？

对于 telemetry counter，“大致接近最近状态”可能就够了；对于授权、billing、ownership 或 replay protection，通常不够。

## 有些 map 应该重建，而不是序列化

BPF map 也不都是普通 key-value storage。[map-of-maps](https://docs.kernel.org/bpf/map_of_maps.html)里保存的是对其他 live map object 的引用；per-CPU map 的状态与 CPU topology 有关；LRU map 把 eviction policy 本身变成语义的一部分；其他 map type 还可能包含 socket、program、queue、ring-buffer 或 kernel-managed relationship。

因此比较实用的做法是给每张 map 明确一种重启策略：

| 策略 | 适用情况 | 重启后动作 |
| --- | --- | --- |
| checkpoint | 这份状态需要保留，而且没有更权威的外部来源 | 通过版本化 state image 和验证 gate 恢复 |
| reconstruct | controller DB、配置、路由表等才是 source of truth | 从权威来源重建并检查 convergence |
| reset | cache、telemetry、临时 queue、boot-local epoch | 明确清空并记录 |
| reject restore | 旧状态无法安全映射到新 kernel/topology/schema | fail closed 或进入显式 fallback |

这比简单问“map 是否 persistent”更接近真正的运维需求。

2026 年 2 月的一个 [Cilium issue](https://github.com/cilium/cilium/issues/44277)也说明，即使没有发生 reboot，pin namespace 与 object identity 已经是生产 control plane 的真实状态机。一次 endpoint regeneration 因为目标路径已有 global BPF-map pin 而进入 recovery loop。这里的问题不是 verifier safety，而是对象身份、pin ownership 与恢复顺序。主机重启会让这个问题再多一层：旧对象身份彻底消失，但 controller 仍然需要重建一套一致状态。

## 现有研究还缺什么

### “持久化”通常没有写清楚到底跨哪条边界

Linux 对 pin 的对象生命周期定义很明确，但应用 manifest 往往不会声明 map 到底要跨 controller restart、kernel reboot、host replacement，还是只跨一次 deployment generation。

缺少的是每张 map 的 lifecycle declaration：它应该声明 survival boundary、durable source of truth，以及恢复证据不足时的 fallback。这样同一个 BPF object 里的两张 pinned map 也可以拥有完全不同的恢复规则。

验证这个 gap 并不难。抽样几个生产 eBPF control plane，如果现有 map declaration 已经能通用地表达 restart boundary、recovery source、semantic schema 与 validation policy，那么再增加一层 contract 的价值就很有限。

### BTF 与 map metadata 不能证明应用语义没有变化

BTF 和 map info 可以描述结构，却不能表达应用层的时间有效性、authority、ownership 或 field meaning。

缺少的是 application-owned semantic state version，并把它绑定到 migrate、reconstruct 或 reset 规则。

最有区分度的测试不是改 value size，而是保持 binary layout 不变，只改变某个 field 的语义。如果恢复系统仍然直接接受旧 image，它还没有解决真正的兼容问题。

### live export 没有通用的 application-level consistency cut

map iteration 和 batch operation 能把数据取出来，但不能自动把并发变化的多张 map 与用户态状态变成同一个 logical snapshot。

缺少的是 quiescence、epoch、copy-on-write 或 delta protocol，用来定义“一份 checkpoint”到底代表哪个时间点。

测试必须在 checkpoint 过程中持续注入 update。如果某个方案只有暂停所有 mutation 才正确，也完全可以接受，只要它明确说这是 stop-the-world checkpoint，而不是把长时间 map walk 当作 consistent snapshot。

### “恢复成功”比“恢复正确”容易检查得多

重新创建 map、插入 entry、load program、attach hook 都成功，不代表恢复后的 policy 或 ownership graph 是正确的。

缺少的是 post-restore semantic gate，以及失败时明确的 reconstruction/reset/fail-closed fallback。

这里最值得测的指标是 **silent bad recovery**。恢复延迟当然重要，但快速恢复一份 stale authorization state 比明确失败更糟糕。

## 兼具学术价值与生产价值的方向

### 1. 给每张有状态 BPF map 一个重启状态契约

**缺口。** map definition 描述怎样创建 kernel object，却没有描述 object 消失之后，应用状态应该怎样跨 reboot。

**机制。** 在 BPF artifact 旁边放一个很小的 manifest，对每张 stateful map 声明 survival boundary、recovery strategy、semantic schema、structural fingerprint、consistency requirement、source of truth 与 restore gate：

```yaml
map: policy_cache
survival: host-reboot
strategy: checkpoint
semantic_schema: policy-state/v3
structural_schema: sha256:<canonical-btf-shape>
consistency: generation-cut
source_of_truth: controller-db
restore_gate: policy-canary/v2
```

structural fingerprint 应该基于 canonicalized BTF shape，而不是 raw type ID；semantic schema 则由应用自己维护，即使 C layout 不变，只要含义改变就要升级版本。

loader 再把这个 manifest 与目标内核的 capability evidence 组合。`reset` map 直接从空状态开始；`reconstruct` 从权威来源重建；`checkpoint` 必须拿到兼容且完整的 state image；证据不够时按 contract 的 fallback 走，而不是猜。

**与现有方案的差异。** 当前 metadata 主要描述 kernel object 与 loader expectation，这个 contract 描述旧 object 已经不存在以后，state 应该如何延续。

**Artifact。** libbpf-compatible manifest library、canonical BTF shape fingerprint，以及 hash、array、per-CPU、map-of-maps 的恢复 adapter。

**评测。** 从 networking、tracing、policy、profiling 应用收集真实 map，注入 structural drift、equal-size semantic drift、CPU topology change、source-of-truth 缺失与 image corruption。测 unsafe restore、unnecessary reset、启动延迟、annotation 成本与 diagnosis time。

**学术价值。** 可以检验 durable BPF state 是否能被压缩成一个小型 type-and-lifecycle contract，而不是每个应用各写一套恢复逻辑。

**生产价值。** 运维人员在 reboot 前就能回答：“哪些状态会回来，为什么？”

**失败条件。** 如果绝大多数生产 map 都只是 disposable cache，或者都能廉价地从外部数据库重建，这套通用 manifest 可能比它解决的问题更重。

### 2. 带一致性语义的 checkpoint protocol

**缺口。** 长时间遍历 map 能复制全部 entry，却可能混合多个 application generation。

**机制。** 给应用增加 checkpoint epoch。checkpoint 开始时 controller 推进 epoch。之后的新 mutation 要么在短暂 quiescence barrier 后执行，要么写入 bounded delta log。用户态先复制 base state，再把 delta drain 到一个明确 cut，最后封存 image 与 cut evidence。

如果 workload 可以接受短暂停顿，同一套接口应提供更简单的 stop-the-world mode。关键不是强行 live checkpoint，而是把使用的 consistency mode 写进结果。

```text
checkpoint_epoch: 8841
base_copy_complete: true
delta_through_epoch: 8863
controller_generation: 8863
consistency_mode: live-delta
```

**与现有方案的差异。** 它不是更快的 `bpftool map dump`，新增的属性是：即使存在并发 update，也能说明这份 image 对应哪个 application-consistent cut。

**Artifact。** userspace checkpoint library、少量 BPF-side epoch/delta helper，以及无法 live logging 时的 stop-the-world fallback。

**评测。** 对高 update rate 的 hash、LRU、per-CPU workload 做 checkpoint，同时注入 writer burst、delete/reinsert、CPU hotplug、controller crash 与 delta-buffer pressure。比较 naive iteration、batch lookup、quiescence、delta logging，测 invariant violation、runtime overhead、pause、lost update、image size 与 recovery point objective。

**学术价值。** 可以观察 database/checkpoint system 的 consistency mechanism 哪些适合 BPF/userspace shared state，哪些会被 map-specific semantic 打破。

**生产价值。** policy 与 accounting 系统不用再假装一次几秒钟的 map walk 等价于瞬间 snapshot。

**失败条件。** 如果短暂 quiescence 已能覆盖几乎所有 reboot workflow，而且成本更低，live delta logging 就应该保持 optional。

### 3. 重新 attach 之前做 staged restore gate

**缺口。** program 能 load、entry 能 insert，并不能证明 reconstructed state 适合新的 kernel、topology 与 application version。

**机制。** 把 restore 当成一次 deployment promotion：

1. 先创建新的 target maps，不接 production hook；
2. 验证 kernel-visible map shape 与 target capabilities；
3. 按 reboot state contract 对每张 map 做 migrate、reconstruct、reset 或 reject；
4. 对准备好的 generation 跑 state invariant 与小型 semantic canary；
5. gate 通过以后再 attach/switch；
6. 失败的 checkpoint evidence 保留足够久，便于诊断。

它可以和 9 月 18 日的 cross-kernel semantic gate 组合。前者验证 program 在新内核上的行为，这里的 restore gate 验证交给 program 的状态本身是否成立。

**Artifact。** VM reboot harness、restore controller，以及包含 valid、stale、partial-write、old-schema、topology-dependent 与 corrupted image 的 state corpus。

**评测。** 在 same-kernel 与 changed-kernel 场景反复 reboot，改变 CPU 数量、map limit、BTF、application/controller version，并在 checkpoint 与 restore 每个阶段 kill process。比较 pin-only restart、naive byte replay、external-source reconstruction 与 gated restore。主要指标是 silent semantic corruption、false rejection、recovery time、state loss 与 diagnosis time。

**学术价值。** 把 BPF reboot recovery 从一组 loader convenience 变成可 falsify 的 correctness property。

**生产价值。** kernel maintenance 与 host replacement 可以拥有明确的 preflight 和 recovery boundary，重要状态出错时能 fail closed。

**失败条件。** 如果普通 load check 与 application health check 已经能捕获所有 injected bad-state case，而且额外 gate 没有带来区分度，就不需要专门机制。

## 现在的生产系统可以怎么做？

第一步是把每张 stateful BPF map 分成 checkpoint、reconstruct、reset 或 reject-on-restore。不要拿“有没有 bpffs pin”来代替这个判断。

第二步是显式维护 semantic state version。BTF 很适合做结构证据，但应用语义需要应用自己拥有版本与 migration rule。

第三步要主动选择 checkpoint 的一致性等级。如果维护窗口允许短暂 quiescence，就用更简单的 pause；如果必须 live checkpoint，再增加 epoch 或 delta mechanism，并且在并发 mutation 下专门测试。

最后，恢复后的 generation 不要立刻接 production traffic。先验证目标内核上的 program，也验证 reconstructed state，再进行 attach 或切换。

## 哪些结果会改变这个判断？

这里假设至少有一部分 eBPF 应用确实需要跨主机维护保存 correctness-sensitive state，而且不能总是从外部 source of truth 廉价重建。如果绝大多数部署都把 BPF map 当作可随时重建的 cache，那么通用 reboot contract 应该保持很轻量，很多应用也不需要 checkpoint system。

如果几乎所有 reboot 都能接受很短的 stop-the-world window，一套复杂 live checkpoint protocol 也没有必要。显式 quiescence 加 semantic schema check 会更简单。

最强的反例，是出现一个成熟且可复用的恢复系统，已经能跨多个互不相关的 eBPF 应用同时处理 per-map durability intent、semantic version、consistent checkpoint cut、topology-aware reconstruction 和 post-restore semantic validation。那会说明这里真正缺的主要是采用率，而不是新的系统抽象。

在那之前，把 pinned map 直接叫作“持久状态”仍然会隐藏最重要的边界：pin 保存的是 live kernel object；主机重启后想安全恢复状态，需要另一套明确的契约。

## 参考资料

- [eBPF Docs：Pinning](https://docs.ebpf.io/linux/concepts/pinning/)
- [Linux kernel documentation：BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation：Hash maps](https://docs.kernel.org/bpf/map_hash.html)
- [Linux kernel documentation：BTF](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel documentation：Map of maps](https://docs.kernel.org/bpf/map_of_maps.html)
- [Linux kernel documentation：Array and per-CPU array maps](https://docs.kernel.org/bpf/map_array.html)
- [libbpf API and source](https://github.com/libbpf/libbpf)
- [Cilium issue #44277：endpoint recovery loop while committing BPF pins](https://github.com/cilium/cilium/issues/44277)
