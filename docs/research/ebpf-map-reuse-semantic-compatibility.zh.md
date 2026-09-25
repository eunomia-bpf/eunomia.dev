---
date: 2026-09-25
slug: ebpf-map-reuse-semantic-compatibility
title: "复用一个 eBPF Map，就代表旧状态还是同一种含义吗？"
description: "eBPF map 参数完全一致，也只能说明内核对象可以机械复用，并不能证明新程序会用同一套 schema 和语义解释旧状态。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
research_question: "eBPF 应用升级后继续复用已有 pinned map 时，需要什么证据才能证明旧状态仍符合新程序预期的结构与语义？"
source_cutoff: 2026-09-25
status: daily-report
---

# 复用一个 eBPF Map，就代表旧状态还是同一种含义吗？

假设 daemon 从版本 N 升级到 N+1。两个版本都定义了同一个 pinned hash map：map type 一样，key size 一样，value size 一样，entry 数量和 flags 也一样。新的 loader 可以重新打开旧 map，新程序也能顺利加载。

这只能证明这个内核对象可以被机械复用，不能证明旧 bytes 在 N+1 里还是原来的含义。

一个 32-byte value 可以保持总大小不变，同时把几个等宽字段调换位置；一个整数的单位可以从“请求数”变成“千分之一 token”；同样宽度的 identifier 也可以从 PID 变成 cgroup ID。所有 lookup 都可能成功，但新程序已经在用错误的 schema 解释真实旧状态。

这个问题比此前的[有状态 eBPF 应用原子升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)更窄。那篇讨论的是整个应用 generation 如何 prepare、migrate、commit、retire；这里问的是单个已有 map 的准入问题：**什么时候可以不迁移，直接复用旧状态？**

它也延续当前 deployment compatibility 系列。此前我们分别讨论了[能力证据](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)、[跨内核语义兼容](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)和[typed interface negotiation](https://eunomia.dev/zh/research/ebpf-kernel-interface-negotiation/)。那些问题关心“代码能不能跑”以及“应该选择哪个 interface variant”；本文关心的是“旧 generation 产生的状态能不能交给新 generation 继续解释”。

<!-- more -->

## Pinning 延长的是对象生命周期，不是应用 schema 生命周期

BPF_OBJ_PIN 会在 BPF 文件系统里保留对内核 BPF object 的引用，因此创建它的进程退出后，对象仍然可以存在。后续进程通过 BPF_OBJ_GET 可以拿到同一个对象的 file descriptor。

这解决的是 object lifetime，不是 schema contract。

因此至少要把两个问题分开：

1. **对象兼容性：** loader 能否获得并使用这个 map？
2. **状态兼容性：** 新旧 producer 和 consumer 是否会用相同的 representation 与应用语义解释现有 key/value bytes？

第一个可以成立，而第二个完全不成立。

Pinning 也不等于持久化数据库。bpffs pin 保存的是正在运行的 kernel 里的对象引用。如果应用要求状态跨 host reboot 存活，还需要独立的 checkpoint/restore 或重建机制。本文关注的是能够实际访问同一个 existing map object 的应用 generation 之间如何安全复用。

## libbpf 今天实际检查什么

当前 libbpf 有两条相关路径。

自动复用已有 pinned map 时，libbpf 会调用 map_is_reuse_compat()，读取 bpf_map_info，然后比较 map type、key size、value size、max entries、map flags 和 map_extra；部分 map type 还会做 flag normalization。参数不匹配时会拒绝自动复用。

这些检查是必要的。一个 hash map 不应该被当成 array 复用，期待 32-byte value 的程序也不应该拿到 24-byte value 的 map。

问题在于，这些字段定义的是 storage shape，不是完整的 application-state contract。比如 N 的 value 依次保存 last_seen_ns、policy_generation、verdict、bytes，N+1 把几个同宽字段重新排序，总 value size 仍然可以完全相同。参数比较全部通过，但 field interpretation 已经不兼容。

显式的 bpf_map__reuse_fd() 更能说明这个边界：当前 libbpf 会读取传入 fd 的 bpf_map_info，并把已有 map 的 type、size、flags、BTF key/value type ID 和 map_extra 等信息带进 libbpf map object。这个 API 的职责是选择一个 existing kernel object，不是证明其旧状态符合新 artifact 的应用语义。

这不是 libbpf 的 bug。低层 loader 不可能从几个 byte count 自动推断任意业务语义。

## BTF 能给出更强的结构证据，但不是 semantic version

BTF 可以暴露更丰富的 representation 信息。bpf_map_info 可以包含 btf_id、btf_key_type_id 和 btf_value_type_id；通过对应 BTF blob，tooling 能看到 member offset、integer encoding、array、struct、union、enum、bitfield 等类型图信息。

这足以区分很多“总 byte size 相同、实际 layout 不同”的情况。

但 raw BTF type ID 不能直接拿来当跨 build schema version。Type ID 是在某个具体 BTF object 的 type section 里分配的局部编号。重新编译后，即使有效 layout 没变，编号也可能变化。因此兼容性规则应该比较经过 normalization 的 reachable type graph，而不是比较整数 ID。

即使结构完全相同，应用语义仍然可能改变。一个 u64 字段可以保持同一个 offset，却把单位从“请求数”改成“milli-request”；一个 u32 可以从 PID 改成 cgroup ID；generation counter 也可能引入新的 epoch 规则。这些都不是 BTF 能自动知道的。

因此，一个可用的 map state contract 至少应该有三层：

- **kernel-visible map definition：** map type、key/value size、entry count、flags、map_extra 与 map-specific constraint；
- **structural schema：** normalize 后的 key/value type graph 与 layout；
- **semantic schema：** unit、identifier namespace、epoch、ownership、reset policy、valid range，以及 C layout 表达不了的 invariant。

## 现有工作还缺什么

### 参数相同仍可能接受错误 decoder

第一层 gap 很直接：只要 key/value 结构变化没有改变 map_is_reuse_compat() 检查的参数，loader 就可能继续接受已有 map。

一个有区分度的 benchmark 应该主动生成这种 mutation：交换等宽字段、改变同大小的 nested struct、修改 enum interpretation、移动 bitfield、改变 key composition，同时保持 kernel-visible map 参数不变。真正要测的是有多少错误 decoder 会被 silent reuse，而不是程序能不能加载。

### Type compatibility 仍证明不了 meaning

更强的 BTF checker 可以抓 layout drift，但抓不到 unit、identifier domain、policy epoch、ownership 等 same-layout semantic drift。

所以必须有 application-declared semantic revision 或 invariant set。否则即使结构 checker 完美，policy、security、accounting、rate limiting、identity 这些高价值状态仍然存在盲区。

最有区分力的实验是故意注入 same-layout semantic mutation。如果机制只会对 BTF 变化报警，它保护的是 ABI，不是 persistent-state meaning。

### Migration 本身还有 consistency cut

如果 direct reuse 被拒绝，migration 也不是离线文件转换。BPF program 可能在 userspace 遍历 hash map 时继续更新；LRU map 会 eviction；per-CPU map 有多个 value slot。没有 quiescence 或 generation boundary，迁移结果可能混入两个 logical epoch 的数据。

所以 schema admission 与 transactional upgrade 最终会在这里相交：前者决定是否必须迁移，后者决定怎样迁移而不丢 concurrent update，并且还能 rollback。

## 兼具学术价值与生产价值的方向

### 方向一：从 BTF 推导稳定 structural fingerprint

**Gap。** Map 参数只能发现 size 和 map shape 变化，raw BTF ID 又不是稳定的跨 build identifier。

**Mechanism。** 从 map 的 BTF key/value type 开始递归 canonicalize reachable type graph，把 type kind、resolved size、member name 与 bit offset、integer encoding、array length、enum representation、nested structural digest 纳入 fingerprint；忽略 BTF-local numeric ID 和不影响 representation 的 build artifact。同时保留 human-readable structural diff。

Compatibility 不应该简单等于 type byte-for-byte identity。工具至少可以支持“严格结构一致”和“显式声明的兼容演进”两种 policy：typedef rename 不必触发 migration，而等宽字段重排应当被拒绝。

**Delta。** 这比 libbpf 当前的 map-definition equality 更强，但明确不冒充 application semantics。

**Artifact。** 一个 libbpf-adjacent 的 map-schema 工具：从 ELF object 提取 expected schema，从 live map 的 BTF 提取 observed schema，生成 canonical digest，并逐字段解释 mismatch。

**Evaluation。** 跨 Clang 版本和 optimization mode 构造 mutation corpus，包括 field reorder、nested type、array、enum、bitfield、padding、typedef-only edit 与 BTF deduplication。测 false accept、false reject、中性 rebuild 下 digest stability，以及 startup cost。

**学术价值。** 把“局部 ID 不稳定的 type metadata 如何定义跨 build representation compatibility”变成可验证问题。

**生产价值。** Loader 可以在状态被错误解释之前，明确告诉 operator 哪个字段发生了什么结构变化。

**失败条件。** 如果普通 toolchain 变化就让 digest 不稳定，或者生产 map 普遍没有可用 BTF，那么 source-generated schema manifest 会是更实际的 fallback。

### 方向二：把 state version 从 bpffs path 中独立出来

**Gap。** Structural equality 表达不了 unit、ownership、reset safety 与 semantic epoch。

**Mechanism。** 为每个 state-bearing map 定义独立逻辑 contract：map identity、structural fingerprint、semantic revision、lifecycle class、可接受的 predecessor revision、migration path 与 reset policy。启动时只能选择四种显式结果之一：direct reuse、migrate、policy-approved reset、hard refusal。

需要 migration 时创建新的 map generation，而不是原地修改旧 representation。先 gate 或 quiesce writer，再 transform entry、验证 invariant、切换 producer/consumer；旧 generation 保留到 rollback window 结束。

**Delta。** 此前 whole-application transactional upgrade 提供 cutover protocol；这里补上的是“哪些 map 能跳过 migration、哪些绝对不能”的状态证据。

**Artifact。** 一个 versioned state manifest 与 migration runner。简单 structural transform 可以自动生成，semantic transform 由应用显式 callback 完成。

**Evaluation。** 在 rolling daemon upgrade 中加入 concurrent writer，并在 migration 每个阶段 crash；覆盖 LRU、per-CPU map 和内存不足以完整复制大 map 的情况。测 lost/duplicated logical update、rollback success、downtime 与可自动迁移比例。

**学术价值。** 把 state compatibility 从 pathname convention 变成可组合的 upgrade property。

**生产价值。** must-preserve map 能得到明确的 refuse/reset/migrate 行为，disposable cache 仍然可以声明 reset-safe。

**失败条件。** 如果绝大多数 pinned state 都能低成本重建，那么复杂 migration machinery 的收益可能小于直接 reset。

### 方向三：给新版本写权限前做 shadow validation

**Gap。** Manifest 也可能写错。两个 release 可以错误声明同一个 semantic revision。

**Mechanism。** N+1 获得 write authority 前，先读取 bounded sample 或 consistent snapshot，检查 declared invariant。对无 side effect 的 decoder，可以让 N/N+1 对相同 raw entry 做 normalize，然后比较 logical record；还可以检查 identifier resolution、counter range、generation membership 与 aggregate parity。

活跃 map 必须带 epoch 或 snapshot boundary，否则 concurrent update 会被误判为 decoder mismatch。验证通过后生成 reuse receipt，绑定 map identity、structural fingerprint、semantic revision、新 artifact identity 与 validation result。

**Delta。** 前两个方向主要依赖部署前生成的 metadata；shadow validation 把真实旧状态本身变成 admission evidence。

**Artifact。** 一个 state-compatibility harness，提供 map-specific sampling adapter 和 application invariant plugin。

**Evaluation。** 注入 same-layout semantic bug、stale declaration、corrupt entry、partial migration 与 concurrent update。比较 manifest-only 与 shadow admission 能抓到多少 silent bad reuse，并测 false alarm 与 startup delay。

**学术价值。** 研究静态 type evidence 不完整时，需要多少 runtime evidence 才能判断 state representation 可复用。

**生产价值。** 高价值 map 在新版本开始 mutation 之前多一道真实状态检查；低风险 map 可以跳过成本。

**失败条件。** 如果应用没有便宜、无 side effect 的 decoder，也没有足够强的 invariant 或一致 observation point，这层机制不应强制全局启用。

## 今天就能落地的 policy

不需要新的 kernel API，loader 现在就可以把两类 admission 拆开：

- artifact + target kernel -> capability / interface admission；
- artifact + existing state -> representation / semantic admission。

具体做法是给 pinned map 标记 ephemeral、reset-safe、must-preserve 或 migrate；先运行现有 map-definition check；有 BTF 时比较 structural schema；must-preserve 状态要求显式 semantic revision；只有 compatibility 成立后才调用 bpf_map__reuse_fd() 完成机械复用；证据不足时 migrate、按 policy reset，或者拒绝升级。

第一个 path 成功，绝不能自动推出第二个也成功。

## 哪些结果会改变这个判断？

如果旧状态本来就明确 disposable，而且能低成本重建，那么更复杂的 contract 没有必要。若所有 consumer 都故意把 value 当 opaque fixed-size byte string，也不需要对内部 source-language layout 做教条式比较。

如果普通 compiler rebuild 就让 BTF-derived fingerprint 不稳定，或者 production map 普遍缺少足够的 BTF，它就不是好的强制机制；source-generated manifest 可能更合适。若真实应用没有强 invariant 或 consistent observation point，shadow validation 的复杂度也可能不值得。

但只要状态必须跨 application upgrade 延续，今天的 parameter check 只能证明 storage object 可以被机械复用。BTF 能提供更强 structural evidence，却仍不能单独证明 application meaning。更安全的边界应该组合 **map-definition compatibility、稳定 structural schema evidence、显式 semantic contract**；三者不一致时进入 migration、reset 或 refusal。

## 参考资料

- [Linux kernel documentation: BPF syscall](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [eBPF Docs: bpf_map__reuse_fd](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_map__reuse_fd/)
- [Eunomia Daily Report: 有状态 eBPF 应用能否原子升级？](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)
