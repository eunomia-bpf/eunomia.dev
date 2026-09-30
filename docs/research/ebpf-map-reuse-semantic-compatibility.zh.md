---
date: 2026-09-30
slug: ebpf-map-reuse-semantic-compatibility
title: "复用的 eBPF Map 还表示同一件事吗？"
description: "libbpf 会在 map 参数不一致时拒绝 pinned map 复用，但字节大小一致并不能证明升级后的程序仍用相同 schema 和语义解释旧状态。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
  - Compatibility
research_question: "当新的 eBPF application generation 复用已有 pinned map 时，什么证据才能证明旧字节仍符合新程序期待的 schema 与语义？"
source_cutoff: 2026-09-30
status: daily-report
---

# 复用的 eBPF Map 还表示同一件事吗？

一个 eBPF daemon 从版本 N 升级到 N+1。两个 generation 都定义了同样的 pinned hash map：map type 相同，key 是 16 bytes，value 是 32 bytes，entry 数量和 flags 也一样。libbpf 可以接受旧对象复用，新程序能够正常 load，每次 lookup 也都会返回预期长度的数据。但这些事实都不能证明 N+1 对这些旧字节的解释与 N 相同。

字段可以换位置而总大小不变；一个 identifier 可以从 PID 改成 cgroup ID 而宽度不变；一个 counter 可以从“请求数”改成“千分之一请求”，C 类型仍然是 `__u64`。Storage object 可以复用，不代表 state contract 也可以复用。

本文的问题比[有状态 eBPF 应用能否原子升级？](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)更窄。之前讨论的是整个 application 的 prepare、migrate、commit、rollback 与 retire；这里关心更早的一道门：**这个旧 map 能不能直接交给新 artifact 解释，还是必须 migration、reset，或者拒绝升级？**

它继续当前的 Deployment Compatibility and Lifecycle 系列，前面的边界包括[能力证据](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)、[跨 kernel 语义兼容](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)和[类型化接口协商](https://eunomia.dev/zh/research/ebpf-kernel-interface-negotiation/)。

<!-- more -->

## 为什么 eBPF map reuse 不是 schema 证明

Linux BPF map 是 kernel object，可同时被 BPF program 与 userspace 使用。把对象 pin 到 bpffs 后，即使创建它的进程退出，后续进程仍然可以通过 filesystem reference 打开同一个活着的 kernel object。这很适合 daemon restart 和滚动 application upgrade。

但 pinning 解决的是 object lifetime，不是 application schema lifetime。普通 bpffs pin 也不会把 kernel object 变成跨主机重启的 durable storage；如果需要 reboot 后继续保留状态，还必须有独立的 serialize/restore 机制。

当前 upstream libbpf 的自动 pinned-map reuse 路径正好体现了这个边界。它先读取 `bpf_map_info`，再把旧 map 与新 definition 做比较。截至本文 source cutoff，`map_is_reuse_compat()` 检查 map type、key size、value size、`max_entries`、map flags 和 `map_extra`，并对 devmap flags 做 map-type-specific normalization。这些检查非常有用，也是必要的 storage-shape gate。

但它们并不是完整 representation check。例如：

```c
/* N */
struct flow_state {
    __u64 last_seen_ns;
    __u32 policy_generation;
    __u32 verdict;
    __u64 bytes;
    __u64 packets;
};

/* N+1 */
struct flow_state {
    __u64 bytes;
    __u32 verdict;
    __u32 policy_generation;
    __u64 last_seen_ns;
    __u64 packets;
};
```

两个 value 的总大小相同；map type、key size、容量、flags 和 `map_extra` 都可以完全一样。因此 parameter equality 无法发现 interpretation 已经变化。

这不是 libbpf 的 bug。低层 loader 不可能从 byte count 自动推导任意 application meaning。

## BTF 能给结构证据，但不是 application semantic version

BTF 让这个问题更可解。Linux 通过 `bpf_map_info` 暴露 `btf_id`、`btf_key_type_id` 和 `btf_value_type_id`；tooling 可以拿到对应 BTF blob，并遍历 reachable type graph。这个 graph 能描述 integer encoding、member offset、array、nested struct、union、enum 等 representation 信息。

但 raw BTF type ID 仍然不能直接当 portable schema version。ID 只在某一个 BTF object 内有意义。重新 build 后，类型编号可以变化而 representation 完全不变；两个独立加载的 BTF blob 也可以给同构类型不同的数字 ID。

即使 structural comparison 完全正确，也不能恢复没有编码在 type graph 里的 application semantics：

```c
struct token_bucket {
    __u64 last_refill_ns;
    __u64 tokens;
};
```

如果 N 把 `tokens` 当“请求数”，N+1 改成 milli-token，BTF 会证明 layout 没变，但 application contract 已经变了。

因此 state reuse 至少需要三层判断：

1. **map-definition compatibility**：kernel-visible storage properties 一致；
2. **structural compatibility**：key/value representation 相同，或符合明确声明的 evolution rule；
3. **semantic compatibility**：单位、identifier namespace、epoch、ownership、validity rule 与 invariant 仍然一致。

`bpf_map__reuse_fd()` 应该放在这三层之下。它负责选用现有 map FD，而不是证明 reuse 正确。

## 现有工作还缺什么

### 相同 map 参数仍可能静默接纳不同结构

最直接缺少的是一个 mutation corpus：保持 libbpf 当前检查的所有参数不变，同时修改 key/value layout。应该覆盖等宽字段重排、nested struct 变化、enum reinterpretation、bitfield 移动以及 key composition 变化。

有用的指标不是“object 能不能 load”，而是 false-accept rate：parameter-only admission 有多少次会允许新程序错误解释旧状态。

### Structural identity 仍然抓不到 semantic drift

更强的 BTF comparison 可以发现 representation drift，但同样抓不到 layout 不变的单位变化、identifier 含义变化、lifecycle epoch 或 algorithm invariant 变化。对必须跨 upgrade 保留的 state，需要额外的 application-declared semantic revision。

区分度高的实验应该故意保持 BTF 完全相同，只改变 meaning。如果机制无法拒绝或迁移这些案例，它保护的是 ABI drift，而不是 semantic drift。

### Migration 本身有 concurrency boundary

当 direct reuse 不合法时，转换 live map 也不是“把一个离线文件转格式”这么简单。BPF program 可能在 userspace iteration 期间更新 entry；LRU map 会 eviction；per-CPU map 有多个 value slot；map-of-maps 又引入一层 identity。

所以 migration 必须定义 old generation 与 new generation 之间的 cut。否则结果可能同时包含 migration 前 entry 与 migration 后 write，却没有清楚 ordering rule。

## 值得做的研究与工程方向

### 方向一：把 BTF type graph canonicalize 成 structural fingerprint

**Gap。** Kernel-visible parameter equality 会漏掉 layout 变化，而 BTF numeric ID 又只在单个 BTF object 内有效。

**Mechanism。** 从 map 的 BTF key/value type 出发，递归 canonicalize reachable representation：kind、resolved size、member name 与 offset、integer encoding/signedness、array length、enum representation，以及 nested structural fingerprint。忽略 BTF-local numeric ID 和不影响 representation 的 build artifact，再对 normalized graph 计算 digest。

工具不应该只支持“完全一致”，还要允许明确声明的 compatible evolution。Typedef rename 不应强制 migration；字段移动应该。Append-only compatibility 只有在真实 consumer 明确声明只读取稳定 prefix 时才应通过。

**与相关工作的差异。** BTF 已经负责携带 type information；这里把它变成跨 build 的稳定 comparison artifact，而不是误把 kernel-local ID 当 version。

**Artifact。** 一个面向 libbpf application 的 `map-schema` 工具，同时输出 machine-readable fingerprint 与 human-readable old/new structural diff。

**Evaluation。** 对 struct、array、enum、padding、compiler version、BTF dedup 与 semantically-neutral rebuild 做系统 mutation。测 false accept、false reject、fingerprint stability 和 startup overhead。

**学术价值。** 核心问题是如何定义“独立生成的 BTF graph 之间的 representation equivalence”。

**生产价值。** Loader 可以明确解释“reuse 被拒绝，因为 `flow_state.last_seen_ns` 从 offset 0 移到了 16”，而不是等 corrupted behavior 出现后再排查。

**失败条件。** 如果 ordinary toolchain rebuild 都会导致 canonical fingerprint 不稳定，或者部署 artifact 没有可用 BTF，那么 source-generated schema manifest 会是更简单的 baseline。

### 方向二：给 state semantics 显式 version，把 migration 变成一等结果

**Gap。** Structural equality 无法描述单位、namespace、epoch 与 application invariant。

**Mechanism。** 给每个 state-bearing map 一个独立于 bpffs path 的 logical contract：

```text
map_identity: flow_state
structural_fingerprint: sha256:...
semantic_revision: 4
lifecycle: must-preserve
migration_from: [2, 3]
reset_policy: forbidden
```

启动时把旧状态分成四种结果：direct reuse、read-and-migrate before write、explicit reset、refusal。如果需要 migration，就创建新 generation map，gate writer，转换并验证 entry，再把 authority 切换到新 generation；旧 pin 要保留到 rollback 条件消失以后。

**与相关工作的差异。** 之前 transactional-upgrade 处理整个 application 的事务协议；这里是用 schema evidence 决定每个 map 的 admission outcome，再把结果交给更大的 upgrade protocol。

**Artifact。** Versioned map manifest + migration runner。简单 structural transform 可以自动生成 adapter，semantic transform 则调用 application callback。

**Evaluation。** 测 rolling upgrade、concurrent writer、每个 migration phase 的 crash、LRU eviction、per-CPU value，以及大到不能便宜复制两份的 map。记录 lost/duplicated update、rollback success、downtime 与 migration cost。

**学术价值。** 问题是 state-schema evolution 能否抽象出一个对不同 map type 都有用的小型 compatibility algebra。

**生产价值。** Operator 可以区分“这个 state 明确允许复用”和“这个 fd 恰好通过低层参数检查”。

**失败条件。** 如果 production map 绝大多数都只是 disposable cache，便宜重建，那么应该优先声明 `reset-safe`，而不是强迫引入 migration machinery。

### 方向三：在给新 generation 写权限前，用旧状态做 shadow validation

**Gap。** Manifest 可能过期或写错，migration callback 也可能生成 structural valid 但 semantically invalid 的 value。

**Mechanism。** 在 N+1 获得 write authority 之前，让它读取 bounded snapshot/sample 并检查 declared invariant：counter 范围、identifier resolution、generation membership、cross-field constraint，以及 old decoder 与 new decoder 对相同 raw entry 的 logical record 是否一致。对 active map，要把 comparison 绑定到 epoch 或 snapshot boundary，避免把正常 concurrent update 错判成 decoder disagreement。

Admission 通过后生成 reuse receipt，绑定 existing map identity、structural fingerprint、semantic revision、新 build identity 与 validation result。

**与相关工作的差异。** Metadata-only compatibility 只看声明；shadow validation 加入“即将被复用的真实旧状态”作为 evidence。

**Artifact。** 一个 compatibility harness，提供 map-type-specific sampling/snapshot adapter 与 invariant plugin。

**Evaluation。** 注入 same-layout semantic bug、stale revision declaration、corrupted entry、partial migration 和 concurrent update。比较 manifest-only 与 shadow validation 抓到的 silent bad reuse、false alarm 与 startup cost。

**学术价值。** 这里的问题是需要多少 sampled behavioral evidence，才能有效区分 representation compatibility 与 semantic compatibility。

**生产价值。** Policy、accounting、security 等高价值 state 在新 writer 获得修改权限前多一道可解释的 guard。

**失败条件。** 如果 application 没有便宜、无 side effect 的 decoder，也没有足够强的 invariant，shadow validation 就应该保持 optional，而不是变成所有 map 的强制仪式。

## 今天就能实现的 eBPF map reuse gate

这个方向不需要等新 kernel API：

1. 把 pinned map 分类成 `ephemeral`、`reset-safe`、`must-preserve` 或 `migrate`。
2. 先做现有 kernel-visible map-definition check。
3. 有 BTF 时，把稳定 structural fingerprint 跟 artifact 一起发布。
4. 对必须跨 upgrade 保留含义的 state，记录显式 semantic revision。
5. 把 `bpf_map__reuse_fd()` 当 admission 之后的 action，而不是 admission proof。
6. 对不确定的 `must-preserve` state 直接拒绝，不要 silent reset 或 blind reuse。
7. Migration 到新 generation，并保留旧 generation 直到 validation 与 rollback 条件满足。
8. 记录 expected/observed schema evidence，让 incident 能重建“为什么当时允许复用”。

这样可以把两类 deployment question 分开：

```text
artifact + target kernel
    -> program/interface admission

artifact + existing map state
    -> representation/semantic admission

两者都通过
    -> attach 并授予 state authority
```

## 什么证据会改变这个结论？

如果 application 本来就把 map value 当 opaque fixed-size byte string，而且两个 generation 使用同一个 opaque protocol，那么 source-language wrapper struct 的变化没有行为意义，这时 richer structural contract 并不必要。

如果 production evidence 表明长期 pinned map 几乎总是两类：要么 ABI 多年冻结，要么只是 disposable cache、每次 upgrade 都能安全重建，那么 parameter check + explicit reset policy 可能已经足够。

如果普通 compiler/toolchain variation 让 canonical BTF fingerprint 在实践里无法稳定，source-generated schema manifest 会更合适。如果真实 map 没有一致 observation point，也没有有用 invariant，shadow validation 的价值也会下降。

因此本文的边界很具体：当前 libbpf check 可以证明已有 map 的 kernel-visible storage shape 与新 definition 兼容；BTF 可以暴露大部分 representation。两者都不能单独证明下一代 application 会给旧 bytes 赋予同一个 meaning。真正需要跨 upgrade 保留的 state，需要一个明确的桥梁，把 **storage compatibility**、**structural schema** 和 **application semantics** 连接起来。

## 参考资料

- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [libbpf API: `bpf_map__reuse_fd`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_map__reuse_fd/)
- [Eunomia Daily Report: 有状态 eBPF 应用能否原子升级？](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)
- [Eunomia Daily Report: eBPF 加载器能相信内核版本号吗？](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)
- [Eunomia Daily Report: eBPF Object 在 Kernel 升级后还能保持原来的含义吗？](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)
- [Eunomia Daily Report: eBPF 加载器能把 kfunc 简化成“有”或“没有”吗？](https://eunomia.dev/zh/research/ebpf-kernel-interface-negotiation/)
- [eunomia-bpf organization](https://github.com/eunomia-bpf)
