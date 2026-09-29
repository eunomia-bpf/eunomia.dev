---
date: 2026-09-29
slug: ebpf-map-reuse-semantic-compatibility
title: "复用的 eBPF Map 还表示同一件事吗？"
description: "libbpf 会检查 pinned map 的复用参数，但 key/value 大小一致并不能证明升级后的 eBPF 程序仍用相同结构和语义解释旧状态。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
  - Compatibility
research_question: "eBPF 应用升级后继续复用 pinned map 时，怎样证明旧状态仍符合新程序期待的结构与语义？"
source_cutoff: 2026-09-29
status: daily-report
---

# 复用的 eBPF Map 还表示同一件事吗？

一个 eBPF daemon 从 N 升级到 N+1。两个版本的 pinned hash map 拥有相同的 map type、key size、value size、entry count 和 flags。libbpf 接受旧对象，新程序也能通过 verifier。

但这不等于旧状态的“意思”没变。N+1 可以在总大小不变的情况下交换同宽字段、修改单位，或者把一个整数改成另一个 namespace 的 identifier。内核看到的 storage shape 没变，应用语义却已经不同。

这个问题比[有状态 eBPF 应用原子升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)更窄。那篇讨论整个应用的迁移协议；这里只问一个 admission decision：**新的 artifact 能不能直接继承这个已有 map？** 它继续当前 deployment compatibility 系列，前面已经讨论了[capability evidence](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)、[跨内核语义兼容](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)和[接口协商](https://eunomia.dev/zh/research/ebpf-kernel-interface-negotiation/)。

<!-- more -->

## Pinning 保住的是对象，不是 application schema

bpffs pin 会保留一个活着的 BPF object 引用。创建进程退出以后，另一个进程还能打开同一个 map。

这解决的是 object lifetime，不是 state compatibility：

```text
object lifetime:      后来的进程还能打开这个 map 吗？
state compatibility:  新代码会正确解释旧 key/value 吗？
```

前者成立，不代表后者成立。普通 pin 也不是跨 host reboot 的 durable storage，内核重启后旧 object 会消失。

## libbpf 现在检查什么

截至 2026-09-29，libbpf 自动复用 pinned map 的 `map_is_reuse_compat()` 会比较已有 map 与新 definition 的 type、key size、value size、`max_entries`、flags 和 `map_extra`。

这些检查很重要，但它们没有描述 field-level layout。例如：

```c
/* N */
struct flow_state {
    __u64 last_seen_ns;
    __u32 generation;
    __u32 verdict;
    __u64 bytes;
};

/* N+1：总大小不变 */
struct flow_state {
    __u64 bytes;
    __u32 verdict;
    __u32 generation;
    __u64 last_seen_ns;
};
```

如果总 size 和 map 参数相同，parameter-level check 无法区分两者。这不是 libbpf 的 bug，而是 application-level state contract 不属于底层 storage API。

## BTF 能证明结构，但不能证明含义

Linux 可以给 map 关联 BTF key/value type，工具也能读取 type graph，比较 member offset、width、nested type、array 和 enum。

但 raw BTF type ID 只在某个 BTF object 内有意义，不适合作为稳定 schema version。更重要的是，完全相同的 layout 也可能有不同 semantics。比如 `__u64 tokens` 从“整数 token”改成“milli-token”，BTF 看不出变化。

因此 persistent-state compatibility 至少有两层：

1. **Structural schema**：map 参数和 key/value 的类型布局。
2. **Semantic schema**：单位、identifier namespace、epoch、合法范围、ownership 等应用 invariant。

`bpf_map__reuse_fd()` 也只是“选择哪个已有对象”的 mechanism，不是 compatibility verdict。

## 现有实践还缺什么

### 相同参数仍可能接受错误 schema

同宽字段重排、nested layout 改动、key composition 变化，都可能保持 libbpf 检查的参数不变。有效的 benchmark 应该专门生成这些 mutation，并测量 silent wrong-state acceptance，而不只是 load failure。

### structural compatibility 仍不等于 semantic compatibility

BTF 无法推断单位、namespace、generation policy 或 cache validity。只检测 layout drift 的方案解决的是 ABI accident，不是 semantic drift。Evaluation 必须加入“布局没变、含义变了”的 case。

### migration 还需要一致性边界

如果不能 direct reuse，逐 entry 转换也不够。BPF program 可能在 userspace 遍历期间继续写 map，LRU 会 eviction，per-CPU map 又有多份 value。Migration 需要明确 old/new generation cut 和 rollback 规则。

## 有学术和生产价值的方向

### 方向一：canonical BTF schema fingerprint

**Gap。** Map 参数抓不到很多 layout drift，raw BTF ID 又不稳定。

**Mechanism。** 从 key/value BTF type 出发，对 reachable type graph 做 canonicalization，把 kind、size、member offset、integer encoding、array length、enum 和 nested fingerprint 编进 canonical form，忽略 build-local ID，再计算 digest。

**Artifact。** 一个 `map-schema` 工具，比较 ELF definition 与 live map，输出 structural diff 和 machine-readable verdict。

**Evaluation。** 自动变异 nested struct、array、enum、padding、compiler version 与 BTF dedup，测 false accept、false reject、digest stability 和 startup cost。

**学术与生产价值。** 研究如何定义跨独立 build 稳定的 persistent-state ABI，并让 operator 能解释为什么 reuse 被拒绝。

**失败条件。** 如果普通 rebuild 都让 fingerprint 不稳定，或生产环境经常没有可用 BTF，就应该改用 source-generated manifest。

### 方向二：显式 semantic revision 与 migration contract

**Gap。** Structural identity 无法编码单位、namespace、epoch 或算法 invariant。

**Mechanism。** 给 state-bearing map 一个 logical identity、structural fingerprint、semantic revision、lifecycle class 和 reset policy。启动时把状态分类为 direct reuse、migrate-before-write、explicit reset 或 refusal。需要 migration 时创建新 generation map，验证后切换，并保留旧 generation 用于 rollback。

**Artifact。** Versioned map manifest + migration runner。简单 structural change 可以自动生成 adapter，semantic change 则调用 application callback。

**Evaluation。** 在 rolling upgrade 中加入 concurrent write、crash、LRU eviction、per-CPU value 和内存压力，测 lost/duplicate update、rollback success 与 downtime。

**学术与生产价值。** 这把 static schema evidence、application semantics 与 failure-atomic state transition 连起来，也让 operator 区分“明确兼容”和“恰好能 reuse”。

**失败条件。** 如果 map 只是便宜的 disposable cache，migration 可能比 reset 更贵，因此 contract 必须允许 `reset-safe`。

### 方向三：写权限前做 shadow validation

**Gap。** Manifest 和 migration callback 都可能写错。

**Mechanism。** 在 N+1 获得 write authority 前，让它读取 bounded sample 或 snapshot，检查 range、identifier resolution、generation membership、cross-field relation，以及 old/new decoder 对同一 raw value 的结果是否一致。Validation 必须绑定 epoch，避免把 concurrent update 当成 decoder disagreement。

**Artifact。** State-compatibility harness，提供 sampling adapter、invariant plugin 和记录 admission evidence 的 reuse receipt。

**Evaluation。** 注入 same-layout semantic bug、stale revision、corrupted entry、partial migration 和 concurrent update，对比 manifest-only 与 shadow validation 的 silent error、false alarm 和 startup latency。

**学术与生产价值。** 研究 bounded runtime evidence 能否补足不完整的 semantic specification，并给高价值 state 一个真实数据上的 admission gate。

**失败条件。** 如果应用没有便宜、无 side effect 的 decoder 或有效 invariant，shadow validation 应保持 optional。

## 今天就能采用的 deployment guidance

Loader 可以马上做七件事：

1. 把 pinned map 分类成 `ephemeral`、`reset-safe`、`must-preserve` 或 `migrate`；
2. 保留 libbpf 的 map-definition check 作为第一层 gate；
3. 有 BTF 时增加 structural fingerprint；
4. semantic revision 单独声明；
5. compatibility 成立后再调用 `bpf_map__reuse_fd()`；
6. direct reuse 不安全时迁移到新 generation；
7. 记录足够 evidence，说明 map 为什么被 reuse、migrate、reset 或拒绝。

对 [eunomia-bpf](https://github.com/eunomia-bpf/eunomia-bpf) 这类 tooling，可以把 program admission 和 state admission 分开。

## 什么证据会推翻这个结论？

如果 production evidence 表明，长期 pinned map 几乎总是 ABI 稳定结构或可安全 reset 的 disposable cache，那么 parameter equality + explicit reset 也许已经够用。

如果 canonicalization 在普通 toolchain 变化下都不稳定，或者部署普遍没有可用 key/value BTF，BTF fingerprint 的收益也会很低。

如果真实 map 没有一致、低成本的 observation point，而且 application invariant 太弱，shadow validation 也无法有效区分正确和错误 decode。

还有一个明确反例：如果两个 generation 都故意把 value 当成 opaque fixed-size bytes，没有 consumer 依赖内部 source-language layout，那么 wrapper struct 的变化并没有行为含义。

因此当前 Linux/libbpf 的 gap 很具体：map parameters 可以证明 storage shape 相容，BTF 可以暴露 structural representation，但它们都不能证明新 application 会给旧 state 赋予同一个 meaning。安全 reuse 应该组合 **map-definition compatibility、稳定 structural evidence 和显式 semantic contract**；三层不一致时进入 migration 或拒绝。

## 参考资料

- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [eBPF Docs: `bpf_map__reuse_fd`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_map__reuse_fd/)
- [Eunomia: 有状态 eBPF 应用能否原子升级？](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)
