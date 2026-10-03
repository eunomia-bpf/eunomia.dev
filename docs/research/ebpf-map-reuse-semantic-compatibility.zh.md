---
date: 2026-10-03
slug: ebpf-map-reuse-semantic-compatibility
title: "复用的 eBPF Map 还表示同一件事吗？"
description: "libbpf 可以在参数一致时复用 pinned eBPF map，但相同大小并不能证明新程序仍以相同 schema 和语义解释旧状态。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
  - Compatibility
research_question: "eBPF 应用升级后继续复用 pinned map 时，应该用什么证据证明旧状态仍符合新程序期待的结构和语义？"
source_cutoff: 2026-10-03
status: daily-report
---

# 复用的 eBPF Map 还表示同一件事吗？

一个 daemon 从 N 升级到 N+1。两个版本定义的 pinned hash map 具有相同 type、key size、value size、entry count 和 flags。libbpf 可以接受旧 map 并直接复用，新程序也可以顺利加载，每次 lookup 都拿到预期长度的数据。

这仍然不能证明旧字节的“意思”没有变。

某个字段可以在总 value size 不变的情况下移动位置。两个版本甚至可以保持完全相同的 C layout，却把一个字段从毫秒改成微秒，或者从一种 identifier namespace 换成另一种 namespace。对 kernel 来说 storage shape 仍然兼容，对 application 来说旧状态却可能已经被错误解释。

这篇报告比此前的[有状态 eBPF 应用原子升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)更窄。那篇讨论 program、link、map 和 controller 的 prepare、migrate、commit、retire。这里只问一个 admission decision：**在新 artifact 获得已有 map 的状态权限之前，什么证据足以证明 direct reuse 是安全的？**

<!-- more -->

## Pinning 保留的是 kernel object，不是 application schema

Linux 的 `BPF_OBJ_PIN` 会通过 bpffs 路径保留对一个 live BPF object 的引用，使它可以超出创建它的 fd 和 process 生命周期继续存在。这是 object-lifetime 机制，不是未来 producer / consumer 对 key/value 语义一致的承诺。

Pinning 也不是 serialization format。跨 reboot durability 需要额外的 persistence / restore 机制。因此 direct reuse 最直接发生在 controller restart、daemon upgrade 和同一次 boot 内的 program generation 切换。

## libbpf 今天到底证明了什么

当前 Linux libbpf 仍通过 `map_is_reuse_compat()` 做自动 pinned-map reuse。在 Linux commit `e767a4ea70a3992c37ed604157d32f0dfbf9b1e3` 中，它比较：

- map type；
- key size；
- value size；
- `max_entries`；
- map flags；
- `map_extra`。

这些检查很必要，它们能拒绝明显的 storage-definition mismatch，但不是完整的 state contract。

Linux BTF 能提供更强证据。`bpf_map_info` 暴露 `btf_id`、`btf_key_type_id` 和 `btf_value_type_id`，工具可以继续取回 BTF blob，检查 reachable type graph。但 numeric BTF type ID 只在某个具体 BTF object 里有意义，不能直接当作跨 build 的 application schema version。

即使 structural comparison 很稳定，也仍然不够。BTF 可以证明 layout 完全相同，却无法知道某个整数的单位从毫秒变成了微秒。

因此 state contract 至少分两层：

1. **structural schema**：map parameters、type graph、field offset、width、signedness、array、enum 和 nested type；
2. **semantic schema**：单位、identifier namespace、epoch、validity rule、ownership 和 application invariant。

`bpf_map__reuse_fd()` 也说明了这条边界。它负责“选用哪个 existing map”，并不负责证明旧 state 对新 artifact 仍然正确。

Production 经验还说明 migration 本身就是独立的 lifecycle surface。Cilium issue #24013 记录过一次 bpffs map migration 被中断后留下 stale `:pending` map，导致后续安装被阻塞。这个案例不是 semantic-schema drift，但它说明 migration、命名、rollback 和 cleanup 都可能在 verifier acceptance 之外独立失败。

## 现有研究还缺什么

### 相同参数仍可能接受错误 representation

Loader 可以满足 `map_is_reuse_compat()` 检查的所有字段，但 key/value layout 已经变化。一个有用的机制需要专门测试 size-preserving mutation：交换同宽字段、改变 nested struct、移动 bitfield、改变 enum representation，同时保持 kernel-visible map definition 不变。

### Structural identity 不能证明 semantic identity

一个字段可以从 PID 变成 cgroup ID 而 representation 完全不变；timeout 可以换单位；cache entry 仍然能读，但在新算法下已经不应该继续相信。因此 evaluation 必须包含“layout 完全一样但语义改变”的 mutation。

### Migration 还需要 concurrency boundary

如果 direct reuse 被拒绝，转换一个 live map 不能当成离线文件转换。BPF program 可以在 userspace iteration 期间继续更新 entry，LRU map 会驱逐 entry，per-CPU map 又有多个 value slot。Migration protocol 必须定义 old/new generation cut 和 rollback rule。

## 有学术价值和生产价值的研究方向

### 1. 从 BTF 派生 canonical structural fingerprint

**机制。** 从 map key/value 的 BTF type 出发，canonicalize reachable graph，再 hash representation 相关事实，包括 kind、size、member name/offset、integer encoding、array length、enum representation 和 nested fingerprint；忽略 BTF-local numeric ID。

**Artifact 与 evaluation。** 做一个 `map-schema` 工具，比较 ELF object 与 existing map，输出 machine-readable verdict 和 human-readable diff。用不同 compiler、BTF dedup、nested type、padding 和 size-preserving mutation 建 corpus，测 digest stability、false accept、false reject 和 startup cost。

**失败条件。** 如果普通 toolchain rebuild 都无法产生稳定 canonical schema，或者 deployment 经常缺少足够的 map BTF，就不应该把它设成 mandatory gate，而应该 fallback 到 source-generated schema manifest。

### 2. 给 state 显式版本和 migration policy

**机制。** 每个 state-bearing map 都有独立于 bpffs path 的 logical identity、structural fingerprint、semantic revision、lifecycle class、允许的 migration source 和 reset policy。Loader 把旧 state 分类为 `direct-reuse`、`migrate-before-write`、`explicit-reset` 或 `refuse`。

需要 migration 时，新建 generation map，gate writer，转换并验证 state，切换 authority，只有新 generation 被接受后才 retire old map。

**Artifact 与 evaluation。** 实现 versioned map manifest 和 migration runner，在每个 migration phase 注入 crash，同时加入 concurrent update、LRU eviction、per-CPU state 和 memory pressure。测 lost/duplicate update、rollback success、downtime，以及多少 migration 可以安全自动生成。

**失败条件。** 如果 workload 的 pinned map 基本都是便宜的 disposable cache，那么 migration 可能比 reset 更贵。Contract 应允许 `reset-safe`。

### 3. 授予 write authority 前做 shadow validation

**机制。** Metadata 也可能写错。N+1 开始写旧 state 之前，让它 decode bounded sample 或 snapshot，再检查 application invariant，或者比较 old/new decoder 产生的 logical record。成功后生成 reuse receipt，绑定 map identity、structural fingerprint、semantic revision、新 build identity 和 validation result。

**Artifact 与 evaluation。** 做一个 state-compatibility harness，注入 same-layout semantic bug、stale declaration、corrupted entry、partial migration 和 concurrent update。比较 manifest-only 与 shadow admission 捕获 silent bad reuse 的能力、false alarm、coverage 和 startup delay。

**失败条件。** 如果 application 没有便宜、无 side effect 的 decoder，也没有有区分力的 invariant，shadow validation 更适合作为 `must-preserve` state 的 risk-based option，而不是 universal requirement。

## 今天就能采用的 production guidance

不需要等待新的 kernel API。Loader 现在就可以把 pinned map 标成 `ephemeral`、`reset-safe`、`must-preserve` 或 `migrate`；先跑现有 parameter gate，再把 structural fingerprint 和 semantic revision 跟 artifact 一起发布；对于无法证明兼容的 `must-preserve` state 直接拒绝。

把 `bpf_map__reuse_fd()` 当作 compatibility 已经成立后执行 reuse 的 mechanism，而不是 verdict。如果必须 migration，在 validation 和 cutover 成功前保留 old generation。

## 哪些结果会改变这个判断？

如果真实 production evidence 表明长期 pinned map 几乎总是永久 ABI-stable，或者只是每次 upgrade 都明确 reset 的 disposable cache，那么 parameter equality 加 reset policy 可能已经足够。

如果 BTF canonical schema 在普通 toolchain rebuild 间不稳定，或者 map type information 经常不可用，BTF fingerprint 价值会下降。如果 workload 没有一致、低成本的 observation point，也没有足够强的 invariant，shadow validation 的价值也会下降。

还有一个明确 counterexample：如果两个 generation 都故意把 value 当成 opaque fixed-size byte string，而且没有 consumer 依赖它的内部 layout，那么 wrapper struct 的变化没有行为意义。Compatibility 应由真实 consumer contract 定义，而不是机械比较 type spelling。

当前 Linux/libbpf 留下的边界很清楚：kernel-visible map parameters 可以证明 storage shape 兼容，BTF 可以暴露大部分 representation，但它们都不能证明新 application generation 仍然给旧字节相同的 meaning。安全的 direct reuse 应同时要求 **map-definition compatibility、稳定的 structural evidence，以及显式 semantic contract**；不一致时进入 migration 或直接拒绝。

## 参考资料

- [Linux kernel documentation: eBPF syscall 与 object pinning](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [Linux source at e767a4ea: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/e767a4ea70a3992c37ed604157d32f0dfbf9b1e3/tools/lib/bpf/libbpf.c)
- [libbpf API surface: `bpf_map__reuse_fd`](https://github.com/libbpf/libbpf/blob/master/src/libbpf.map)
- [Cilium issue #24013: bpffs map migration 的 stale state](https://github.com/cilium/cilium/issues/24013)
- [Eunomia Daily Report: 有状态 eBPF 应用能否原子升级？](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)
