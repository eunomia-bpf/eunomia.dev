---
date: 2026-09-19
slug: ebpf-pinned-map-reboot-state
title: "Pinned eBPF Map 能跨主机重启保住状态吗？"
description: "bpffs pin 能让 BPF map 跨进程重启继续存在，却不能跨主机重启保存内核对象。本文讨论如何一致地 checkpoint、重建并验证 eBPF 状态。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - BPF Maps
  - State Recovery
  - BTF
research_question: "当 bpffs pin 只能延长内核对象生命周期，而不能跨内核或主机重启保存对象时，eBPF 应用需要怎样的状态契约，才能可靠保存或重建 map 状态？"
source_cutoff: 2026-09-19
status: daily-report
---

# Pinned eBPF Map 能跨主机重启保住状态吗？

一个 eBPF daemon 创建 hash map，把 policy 状态写进去，再 pin 到 `/sys/fs/bpf`。daemon 退出以后，新进程仍然可以重新打开同一个 map，继续用原来的数据。

这很容易让人把 pinned map 叫成“持久化状态”。

但如果机器重启，情况就完全不同。

Linux 对 `BPF_OBJ_PIN` 的定义是：在 BPF filesystem 里保留一个指向现有 BPF object 的引用，使原始 fd 关闭以后对象仍不会被释放。这个机制解决的是**内核对象生命周期和某个 userspace 进程生命周期不一致**的问题。它没有把 map 内容变成一个可以在新内核启动后重新加载的磁盘数据库。

所以至少要把四种场景分开：daemon 重启、agent 升级、kernel reboot、host replacement。pin 对第一种非常有效，对第二种常常也有帮助；到了后两种，仅靠 pin 不够。

真正困难的地方也不是“把 map dump 成文件”。一个还在工作的 map 可能一边被 BPF 程序修改，一边被 userspace 读取；libbpf 的 pinned-map reuse 主要检查内核可见的结构属性；BTF 能描述类型，却不能自动告诉你字段的业务含义；而有些 map 状态从一开始就更适合重新构造，而不是把旧字节原样搬回来。

本文因此聚焦一个比 [stateful eBPF transactional upgrade](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/) 更窄的边界。之前的问题是活着的程序、link、map 和 controller 怎么协调切换。这里假设旧 kernel object graph 已经不存在，我们要回答的是：**凭什么相信新启动的系统恢复出了正确的状态？**

<!-- more -->

## Pin 保住的是内核对象引用，不是 durable state

Linux UAPI 对 `BPF_OBJ_PIN` 的描述很清楚。pin path 会持有 BPF object 的引用，因此创建对象的进程关闭 fd 后，对象仍然可以存在；`unlink()` 删除 pin 后，这个引用消失；当 fd、pin 和其他引用全部消失，对象才会释放。

这个语义很有用。它允许 loader 自己崩掉而 datapath 继续运行，也允许新的 controller 用 `BPF_OBJ_GET` 重新取得 fd，还能让多个工具共享同一个 map。

但这仍然是同一个正在运行的 kernel 里的 reference-counting 规则。

机器重启以后，旧 kernel 中的 BPF object 已经不存在。常用的 eBPF pinning 文档也直接说明：pin 是 ephemeral 的，不会跨 system restart 持久存在。

因此更准确的生命周期应该写成：

```text
进程生命周期 < pinned BPF object 生命周期 < kernel / host 生命周期
```

如果只写“persistent map”，却不写清楚 persistent 到哪一层，就很容易把“daemon crash 后还能继续用”误解成“reboot 后数据还能回来”。

即使 `/sys/fs/bpf` 在开机时重新 mount，也只是重新建立 bpffs 这个伪文件系统。它不会凭空重新创造旧 map 和旧 entries。

## Reuse 一个活着的 pinned map，和 restore 一个新 map，不是同一件事

libbpf 已经有成熟的 pinned-map reuse 路径。loader 可以打开现有 pin，通过 `BPF_OBJ_GET_INFO_BY_FD` 读取 map 信息，检查它和新 BPF object 里声明的 map 是否兼容，然后让新程序继续使用那个 fd。

当前 libbpf 的 compatibility check 会看 map type、key size、value size、`max_entries`、map flags、`map_extra` 等内核属性。这类检查非常合理，因为 loader 必须先保证新程序不会拿一个完全不同形状的 map 来用。

但它不是 durable-state schema system。

例如下面两个 value 都是 8 bytes：

```c
struct state_v1 {
    __u32 verdict;
    __u32 generation;
};

struct state_v2 {
    __u32 verdict;
    __u32 lease_seconds;
};
```

尺寸完全一样，旧数据却不能因为“能塞进去”就被解释成新语义。

BTF 可以把这个问题往前推进。Linux 能给 map 关联 key/value 的 BTF type，并通过 `BPF_OBJ_GET_INFO_BY_FD` 暴露相关 metadata。工具因此可以看到结构体、字段和类型，而不只是字节数。

但 BTF 仍然不能替应用决定：`generation = 7` 能不能变成 `lease_seconds = 7`，counter 是应该继承还是清零，旧的 authorization cache 在 reboot 后是不是仍然有效。

所以 reboot restore 至少需要两层兼容性：

```text
kernel / map compatibility
    type + size + flags + map-specific constraints + target support

application-state compatibility
    schema version + field meaning + ownership + validity + migration rule
```

普通 loader 天然更擅长第一层，第二层必须由应用或 runtime 明确声明。

## 把 map 全部读出来，不等于拿到了一个一致 checkpoint

即使是最普通的 hash map 或 array map，也还有一个经常被忽略的问题：checkpoint 的同时，BPF 程序可能还在写。

Linux 明确允许不同 CPU 上的 BPF program 并发访问 hash-map value。userspace 可以用 `bpf_map_get_next_key()` 遍历，也可以用 batch lookup 加速读取。但是这些 API 解决的是访问和遍历，不会自动把一个正在变化的 map 变成 multi-entry transaction snapshot。

假设状态里有两个相互关联的记录：

```text
A: policy generation = 42
B: generation 42 对应的 allowed principals
```

checkpoint 先读到 A，然后系统把 A、B 都切到 generation 43，最后 checkpoint 才读 B。最终落盘文件可能组合出了一个逻辑上从未存在过的状态。

单个 value 里的 `bpf_spin_lock` 也不能自动解决整个应用的一致性。真正需要保护的往往是很多 key、多个 map，甚至 userspace controller 自己的状态。

对于统计 counter 或 best-effort telemetry，这种近似快照也许完全够用。但对于 policy、ownership、recovery cursor 或 resource accounting，就必须写清楚 consistency boundary。

因此核心问题不是“能不能把所有 key dump 出来”，而是：

> 恢复以后必须满足什么 invariant？checkpoint 过程中又用什么协议证明这个 invariant 对应同一个 recovery cut？

## 并不是所有 BPF map 都应该序列化

通用 checkpoint 还有一个陷阱：把所有 map 都当成 portable bytes。

BPF map 的语义差异很大。per-CPU map 的 value 分散在 CPU 上；map-of-maps 保存的是其他 map object 的引用；LRU map 把淘汰行为本身作为语义的一部分；还有 queue、stack、ring buffer、socket/program reference，以及其他和当前 kernel object 或 device 状态强相关的 map 类型。

一个现实应用里，不同 map 的 reboot policy 可以完全不同：

- **checkpoint**：policy 或 learned state，确实需要保留具体值；
- **reconstruct**：可以从 controller database、配置、routing table 或其他 source of truth 重建；
- **reset**：cache、telemetry、transient queue、boot-local epoch 等本来就只在当前 boot 有意义；
- **reject restore**：旧 representation 在新 kernel、新 topology 或新 schema 下不能安全解释。

如果 runtime 默认为“所有 map 都 dump 一遍，开机再写回去”，很容易得到一个看起来完整的 backup，却没有证明它可以安全 restore。

2026 年 2 月的 Cilium issue #44277 虽然不是 reboot 故障，但很能说明 bpffs pin 其实属于控制平面 state machine。endpoint regeneration 在提交 global BPF map pin 时遇到已有 path，随后进入 recovery loop。这个例子说明 object identity、pin ownership、create/reuse 决策和 cleanup ordering 都会参与真实恢复流程。reboot 只是在这个状态机里再增加一个更彻底的边界：旧 object identity 已经无法继续存在。

## 现有研究还缺什么

### “Persistent map” 很少明确 persistent 到哪一层

Linux 对 object lifetime 的定义其实很清楚，成熟 loader 也知道如何 reopen/reuse pin。薄弱的是应用层契约。很多系统说某个 map 是 persistent，却不区分 process restart、agent replacement、kernel reboot 和 host replacement。

缺少的是 map-level restart contract：明确 state 应该活过哪一种 failure/restart boundary，以及 durable source of truth 在哪里。

如果调查一批生产 eBPF control plane 后发现它们已经用统一、机器可读的方式声明 reboot durability、reconstruction source、semantic schema 和 restore validation，那就不需要再造一层新 contract。

### 现有 compatibility check 主要还是 structural

libbpf 可以拒绝 type/size/flags 不匹配的 pinned map，BTF 还能补充 key/value 的类型结构。但这些都不能证明 replay 后值的业务含义仍然一致。

缺少的是独立于 byte layout 的 semantic state version，以及对应 migration/reconstruction rule。

最有区分度的测试不是“把 value 加 8 bytes”，而是故意做 equal-size semantic drift。如果系统只会抓住尺寸变化，却会错误接受字段含义已经变化的旧数据，它还没有解决核心问题。

### Live map export 没有通用的 application-level snapshot boundary

lookup/batch API 可以把数据读出来，但多个 map 可能一直在并发更新。kernel 不知道哪些 entry 必须属于同一个应用 epoch。

缺少的是 quiescence、epoch 或 delta protocol，把大量独立读取绑定成一个明确的 recovery cut。

评测必须在 checkpoint 过程中持续注入写入。如果某个方案只能先停掉所有 BPF program 再 copy，也可以是很好的工程方案，但应该明确叫 stop-the-world checkpoint，而不是宣称 live consistency。

### Restore 成功通常验证的是“能加载”，而不是“状态正确”

新 map 可以创建成功，旧 entry 也可以全部 insert 成功，BPF program 甚至能顺利 load/attach，但 policy、ownership graph 或 counter 仍然可能已经错了。

缺少的是 post-restore semantic gate，用 workload-specific invariant 判断恢复后的状态是否值得投入生产。

真正应该统计的是 silent bad recovery，而不只是 restore latency。

## 兼顾学术价值和生产价值的方向

### 1. 给每个 stateful BPF map 一个 restart contract

**缺口。** Pin 只说明 live object 能活过 fd 或 controller process，却没有说明 map data 是否应该跨 reboot、重启后从哪里恢复、使用哪个 semantic schema。

**机制。** 在 BPF application artifact 旁边增加一个很小的 state manifest，对每个 map 声明：

```text
map: policy_cache
survival: reboot
strategy: checkpoint | reconstruct | reset
schema: policy-state/v3
kernel_shape:
  type: HASH
  key_btf_digest: ...
  value_btf_digest: ...
consistency: generation-cut
source_of_truth: controller-db
restore_gate: policy-canary-v2
```

BTF fingerprint 应该基于 canonicalized structural description，而不是直接保存 raw BTF type ID，因为 type ID 只在对应 BTF object 内有意义。另一方面，`schema` 是应用自己的 semantic version，即使 C struct 完全没变，只要含义变化也必须升级。

loader 再把这个 manifest 和 target kernel capability evidence 结合起来。`reset` map 创建空表；`reconstruct` map 从 source of truth 重建；`checkpoint` map 则必须拿到兼容 image，并通过 restore gate。

**和现有机制的差别。** 普通 map definition 说明“如何创建 kernel object”。restart contract 说明“这个 object 消失以后，state 应该怎么继续”。

**原型。** 做一个兼容 libbpf 的 manifest parser、canonical BTF schema fingerprint library，再实现 hash、array、per-CPU、map-of-maps 和若干不可直接序列化 map 的 adapter。

**评测。** 从真实 networking、tracing、policy 应用里选 20 到 30 个 map，注入 structural drift、equal-size semantic drift、CPU/topology 变化和 source-of-truth 缺失。测 unsafe restore、unnecessary reset、需要人工 annotation 的数量、startup latency，以及 recovery mode 自动选择正确率。

**学术价值。** 可以验证 BPF state durability 是否能被压缩成一个小型 type/lifecycle contract，而不需要每个项目自己写完全不同的恢复逻辑。

**生产价值。** 运维在 reboot 前就能回答“哪些 state 会回来、哪些会丢、为什么”，也能审计某个 map 为什么选择 restore 而另一个选择 rebuild。

**失败条件。** 如果绝大多数生产 map 不是 disposable cache，就是已经完全由 external database 管理，那么通用 manifest 可能比真正需要保护的状态更复杂。

### 2. 为仍在运行的 map 做 quiescence-aware checkpoint

**缺口。** Batch lookup 可以快速读 map，却没有定义 BPF 和 userspace 并发写入时的逻辑一致 cut。

**机制。** 引入 application checkpoint epoch。checkpoint 开始时 controller 推进一个 BPF program 可见的 epoch。之后发生的 mutation 要么带上新 epoch，要么 mirror 到一个有界 delta log。userspace 用 batch lookup copy base map，记录 checkpoint epoch，再 drain 并应用 delta，直到一个明确 cut，然后才 seal image。

如果应用允许短暂停顿，同一套接口可以走更简单的 quiescence barrier：阻止新 mutation，按照 hook 语义等待 in-flight operation 完成，copy 后再恢复。重点不是所有场景都用 delta log，而是 checkpoint 必须告诉 operator 自己采用了哪一种 consistency mode。

持久化 image 同时保存 recovery-cut evidence，例如：

```text
checkpoint_epoch: 8841
base_copy_complete: yes
delta_range: 8841..8863
delta_complete: yes
controller_state_generation: 8863
```

**和现有机制的差别。** 这不是一个“更快的 map dump”，而是把 concurrent mutation 正式纳入 checkpoint correctness。

**原型。** userspace checkpoint library + BPF-side epoch/delta helper，提供 ring buffer 或 side map 模式，以及 stop-the-world fallback。

**评测。** 在高更新率 hash、LRU、per-CPU workload 上持续 checkpoint，注入 burst write、CPU hotplug、entry delete/reinsert、controller crash 和 delta-buffer pressure。测 invariant violation、checkpoint pause、runtime overhead、image size、lost update 和 RPO。

**学术价值。** 可以研究 database/checkpoint 里的 consistency 方法放到 kernel/userspace shared state 后，哪些仍然成立，哪些会被 map-specific semantics 打破。

**生产价值。** Stateful policy 和 accounting 系统不需要再把一次很长的 map walk 当成“某个瞬间”的完整状态。

**失败条件。** 如果绝大多数 reboot workflow 都可以接受一次很短的 stop-the-world pause，那么 live delta protocol 应该只是可选模式，而不是默认基础设施。

### 3. 把 reboot restore 做成一个经过验证的 deployment gate

**缺口。** reboot 后重新 create map、插入 entry、load program、attach hook 全部成功，也不等于恢复状态在新 kernel、新 topology 上仍然正确。

**机制。** 把 restore 拆成 staged promotion：

1. 创建 fresh target maps，但先不接入 production hook；
2. 验证 kernel-visible shape 和 canonical BTF schema；
3. 按 restart contract 对每个 map 执行 migrate、reconstruct、reset 或 reject；
4. 对 prepared generation 运行 state invariant 和小型 semantic canary；
5. gate 通过后才 attach/switch 到 production；
6. 保留 checkpoint 和失败 evidence 一段时间，不要成功启动后马上覆盖掉诊断信息。

这和 9 月 18 日的 [cross-kernel semantic compatibility](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/) 是互补关系。前者问同一个 artifact 换 kernel 后行为是否一致；这里问的是：**旧 kernel state 已经消失后，新构造出来的 state 是否仍然是这个 artifact 的合法输入。**

**原型。** VM reboot harness、restore controller，以及一组 valid、stale、partial-write、old-schema 和故意 corrupt 的 state images。

**评测。** 覆盖 same-kernel reboot 和 changed-kernel reboot，改变 CPU 数量、map limit、feature support、schema version 和 controller version。对 checkpoint writer 与 restore controller 的每个阶段做 kill injection。比较 pin-only restart logic、naive byte dump/replay、external-database reconstruction 与 gated restore。主要指标是 silent semantic corruption、false rejection、recovery time、state loss 和 diagnosis time。

**学术价值。** 把 BPF restart recovery 从 loader 的 best effort 行为变成一个可证伪的 correctness problem。

**生产价值。** Kernel maintenance 可以拥有明确 preflight/recovery contract，而不是开机后才发现所谓 persistent state 到底哪些只是 kernel-local volatile state。

**失败条件。** 如果 external authoritative store 配合普通 startup logic 已经能以很低 correctness risk 和可接受时间完整恢复，那么 application-specific reboot gate 就够了，不需要一个共享 BPF 层。

## 实际部署应该把不同证据分开

并不是每个 map 都值得 checkpoint。一条更现实的 recovery ladder 是：

```text
只是 process restart，kernel 没变？
    -> structural + semantic contract 匹配时 reopen/reuse live pin

kernel 或 host restart？
    -> 原 pin 对应的 object reference 已不存在
    -> 重建 map graph

strategy = reconstruct？
    -> 从 external source of truth 重建

strategy = checkpoint？
    -> 验证 image + schema + recovery cut
    -> restore 到 fresh map

strategy = reset？
    -> 从空状态开始，并明确记录 continuity 被主动放弃

prepared generation 通过 invariant/canary？
    -> attach/promote
    -> 否则 fail closed、重新 reconstruct，或进入声明过的 degraded mode
```

这样每一个词都有明确含义。`pinned` 只说明 object 可以活过 controller-process lifetime；`checkpointed` 说明某个 consistency cut 有 durable image；`restored` 说明 image 或 source of truth 构造出了新的 map graph；`validated` 才表示应用检查了让这份 state 有意义的 invariant。

这些 guarantee 不应该混成一个“persistent”标签。

对于 [eunomia-bpf](https://eunomia.dev/zh/eunomia-bpf/) 这样的 eBPF toolchain，以及它的 [GitHub repository](https://github.com/eunomia-bpf/eunomia-bpf)，比较自然的落点是 application package / loader manifest：BPF object 除了携带 map creation metadata，还同时声明 state lifecycle。

## 什么证据会改变这个结论？

有三种结果会削弱 first-class reboot-state contract 的必要性。

第一，如果生产 eBPF 应用调查发现绝大多数 state 要么可以直接丢弃，要么已经完全从 external database 重建，那么 reboot durability 应该留在各自 controller，而不是再抽象一层通用 BPF runtime。

第二，如果 structural map properties 加 canonical BTF shape 已经能在真实升级中非常准确地预测 restore correctness，几乎看不到 equal-layout semantic failure，那么额外 semantic schema layer 可以大幅简化。

第三，如果维护场景普遍可以低成本 quiesce BPF mutation，live checkpoint consistency 很少有实际需求，那么通用 epoch/delta protocol 会变成没有必要的 runtime overhead。

现有机制支持的是更克制的结论：Linux pinning 很适合把 BPF object lifetime 和 userspace process lifetime 解耦，但它不是 durable storage。一旦恢复需要跨越 kernel-lifetime boundary，可靠系统就必须明确回答四件事：**哪些 state 真正需要 durable、怎么得到一致 checkpoint、怎么给状态语义做 versioning、以及用什么证据证明重建后的 state 可以安全上线。**

## 参考资料

- Linux UAPI `BPF_OBJ_PIN` / `BPF_OBJ_GET` 生命周期语义：<https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h>
- Linux BTF 文档，包括 map key/value BTF metadata 与 `BPF_OBJ_GET_INFO_BY_FD`：<https://docs.kernel.org/bpf/btf.html>
- Linux BPF hash map 文档，并发访问与 userspace iteration：<https://docs.kernel.org/bpf/map_hash.html>
- Linux bpftool map 文档，pin、map create、inspection 与相关操作：<https://github.com/torvalds/linux/blob/master/tools/bpf/bpftool/Documentation/bpftool-map.rst>
- libbpf source 与 pinned-map reuse compatibility logic：<https://github.com/libbpf/libbpf/blob/master/src/libbpf.c>
- eBPF Docs pinning concept，对 system restart boundary 的说明：<https://docs.ebpf.io/linux/concepts/pinning/>
- Cilium issue #44277，2026-02-10，生产环境中已有 global BPF map pin 导致 regeneration recovery failure：<https://github.com/cilium/cilium/issues/44277>
