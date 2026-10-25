---
date: 2026-09-23
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
research_question: "eBPF 应用升级后继续复用 pinned map 时，应该用什么证据证明旧字节仍符合新程序所期待的结构和语义？"
source_cutoff: 2026-09-23
status: daily-report
---

# 复用的 eBPF Map 还表示同一件事吗？

一个 daemon 从版本 N 升级到 N+1。两个版本都定义了同一个 pinned hash map：map type 一样、key 都是 16 字节、value 都是 32 字节，`max_entries` 和 flags 也没有变。libbpf 打开旧的 pinned map，认为参数兼容，于是直接复用。

新程序顺利加载，verifier 也接受了。每次 lookup 都能拿到预期长度的数据。

这仍然不能证明旧状态的“意思”没有变。

N 版本也许把 value 的前 8 字节解释成 `last_seen_ns`，后面 4 字节解释成 policy generation；N+1 完全可以在总大小不变的情况下交换字段位置、改变 signedness、把某个字段从一种单位换成另一种单位，甚至把一个 32-bit identifier 换成另一个同宽但完全不同 namespace 的 identifier。对内核来说，map 仍然是一个合法的 32-byte value map；对应用来说，旧状态已经可能被新程序错误解释。

这篇报告比此前的[有状态 eBPF 应用原子升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)更窄。那篇讨论 program、link、map 和 controller 的 prepare / migrate / commit / retire 整体协议。这里关注的是**一个已有 map 能不能直接复用的 admission decision**：在 N+1 获得旧状态的读写权之前，到底应该证明什么？

它也继续当前的 deployment compatibility 系列：此前分别讨论了[真实 host capability evidence](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)、[跨 kernel 的行为语义兼容](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)以及[typed interface negotiation](https://eunomia.dev/zh/research/ebpf-kernel-interface-negotiation/)。那些问题回答“代码能不能运行、应该选择哪个接口变体”；今天的问题是：**旧 application generation 创建的状态，能不能被新 generation 继续解释。**

<!-- more -->

## Pinning 延长的是 kernel object 生命周期，不是 application schema 生命周期

Linux 通过 `BPF_OBJ_PIN` 和 `BPF_OBJ_GET` 让 BPF object 可以挂到 bpffs 路径上。创建进程退出后，pin 仍然持有引用；后来的进程可以重新打开同一个 kernel object，并继续使用里面的内容。

这是 object lifetime 机制，不是未来 application schema 的兼容承诺。

还有一个容易混淆的边界：普通 bpffs pin 本身并不是跨机器重启的 durable storage。正常 reboot 后，原来的 kernel object 不会因为 `/sys/fs/bpf/...` 这个名字就自动恢复。应用如果需要跨 reboot 保存状态，还必须有额外的 serialize/restore 机制。因此这里最直接的场景是同一次 boot 内的 controller restart、daemon upgrade、program reload 或 application generation 切换。对于从外部恢复回来的 map state，同样存在 schema 问题，但“恢复”本身是另一层机制。

所以要分清两个完全不同的问题：

```text
object lifetime
    “另一个进程还能不能拿到这个 map 的 fd？”

representation compatibility
    “新 producer / consumer 会不会正确解释已有 key/value 字节？”
```

前者成立，不代表后者成立。

## libbpf 已经检查 map 参数，但这些参数还不是完整状态契约

当前 libbpf 的自动 pinned-map reuse 路径里，`map_is_reuse_compat()` 会读取已有 map 的 `bpf_map_info`，并比较 map type、key size、value size、`max_entries`、flags 和 `map_extra`。个别 map type 还有对应的 normalization。

这些检查非常必要。新对象期待 array，却拿到一个 hash；或者新程序期待 32-byte value，旧 map 只有 24 bytes，都应该尽早失败。

但它检查的是 kernel-visible map definition，而不是应用 schema。看下面两个 value：

```c
/* version N */
struct flow_state {
    __u64 last_seen_ns;
    __u32 policy_generation;
    __u32 verdict;
    __u64 bytes;
    __u64 packets;
};

/* version N+1：总大小不变，但解释不兼容 */
struct flow_state {
    __u64 bytes;
    __u32 verdict;
    __u32 policy_generation;
    __u64 last_seen_ns;
    __u64 packets;
};
```

两个 struct 完全可以得到同样的 `value_size`。map type、key size、entry count、flags 和 `map_extra` 也都可以一样。因此 parameter-level reuse check 无法区分它们。

这不是 libbpf 的 bug。libbpf 不可能从几个 byte count 推断任意 application semantics。真正缺的是 persistent map state 周围的 application/tooling contract。

## BTF 能提供更强的类型证据，但 raw type ID 不是 schema version

BTF 让这个问题变得更可解。BPF map 可以带 key/value type information；Linux 的 `bpf_map_info` 也会暴露 `btf_id`、`btf_key_type_id`、`btf_value_type_id`。工具可以继续取回对应 BTF blob，看到完整的 type graph。

这比“`value_size == 32`”强很多。

但是不能直接把数值 BTF type ID 当作跨版本 schema ID。一个 type ID 只在某个具体 BTF object 内有意义；重新编译后，即使 C 类型完全没变，type 编号也可能因为 dedup 或其他类型集合变化而重新分配。反过来，仅仅保持 type name 不变，也不能证明 field offset 和 field meaning 没变。

因此 reuse decision 需要的是一个稳定的“相关 type graph 表示”，而不是 kernel-local integer ID。

而且 structural identity 仍然不是终点：

```c
struct token_bucket_v1 {
    __u64 last_refill_ns;
    __u64 tokens;
};

struct token_bucket_v2 {
    __u64 last_refill_ns;
    __u64 tokens; /* 现在单位从 token 改成 milli-token */
};
```

这两个版本的布局可以完全相同。BTF 能证明 shape，却不知道 comment 里那个单位变化，更不知道“一 token 是否代表一次 request”这种应用 invariant。

因此 persistent-state contract 至少有两层：

1. **structural schema**：type graph、field offset、width、signedness、array、nested type、enum representation 和 map parameters；
2. **semantic schema**：单位、identifier namespace、lifecycle epoch、合法范围、ownership rule，以及 C layout 里根本没有编码的应用 invariant。

## `bpf_map__reuse_fd()` 更说明兼容判断必须由上层显式负责

libbpf 还提供 `bpf_map__reuse_fd()`，让 loader 把一个已有 map fd 显式关联到 BPF object 里的 map。自己管理 map 生命周期、跨 object 共享状态时，这个 API 很实用。

但它不应该被理解成 compatibility proof。它解决的是“使用哪个已有 kernel object”，不是“这个 object 对新 artifact 是否语义正确”。

这种职责分离本身是合理的：底层 library 不应该假装理解业务语义。代价是 production loader 必须有明确 policy，不能把 reuse fd 成功当作“状态兼容”的证据。

## 现有实践还缺什么

### 相同 map 参数仍然可能接受错误 schema

第一个 gap 很机械：旧 map 和新 map definition 的所有 kernel-visible 参数都相同，但 key/value 的 BTF structure 可以在保持总 size 的情况下变化。

最直接的实验是自动生成一组 schema mutation，同时保持 `map_is_reuse_compat()` 检查的参数不变：交换同宽字段、改变 nested struct、移动 bitfield、改变 enum interpretation、改变 key composition。然后测试哪些 mutation 能通过 parameter-only reuse，以及哪些错误最后表现成 silent behavior bug 而不是 load failure。

一个真正有价值的机制，必须拒绝这些 wrong-schema case，同时不能因为普通 rebuild 里 BTF 编号变化就误拒绝语义完全相同的 map。

### structural compatibility 仍然无法证明 semantic compatibility

更强的 schema checker 也不能自动理解应用含义。32-bit field 可以从毫秒变成微秒；一个整数可以从 PID 变成 cgroup ID；generation counter 的使用规则可以变化；cache value 的字节都能读，但新算法已经不应该继续信任旧缓存。

所以 structural evidence 之外，还需要 application 声明的 semantic revision 或 invariant set。

Evaluation 必须故意包含“布局完全不变、语义发生变化”的 mutation。如果一个方案只会抓 layout drift，它解决的是 ABI accident，不是 persistent-state semantic drift。

### live migration 还有 concurrency boundary

即使 N 和 N+1 都知道怎样转换一个 entry，把活跃 BPF map 做 migration 也不是转换一个离线文件。BPF program 可以在 userspace 遍历期间继续更新 entry；LRU map 可以驱逐 entry；per-CPU map 每个 CPU 都有 value；map-in-map 和一些 object reference map 又有额外生命周期；某些 value 还带 timer 或 spin lock 约束。

因此 migration 需要一个 old/new generation cut。否则 migration 结果可能混合“迁移前 entry”和“迁移期间新 update”，却没有清楚 ordering semantics。

这里正好连接回此前 transactional-upgrade 工作：schema admission 决定能否 direct reuse；如果不能，真正 migration 仍需要 generation control 和 rollback。

## 方向一：从 BTF type graph 计算稳定 structural fingerprint

**Gap。** Map parameter equality 能发现 size 和 map shape 改变，却发现不了很多保持 size 不变的 key/value layout drift；raw BTF type ID 又不能跨独立 BTF object 稳定比较。

**Mechanism。** 从 map 的 key/value BTF type 出发，对 reachable type graph 做 canonical normalization，再计算 fingerprint：

```text
kind
  + resolved size/alignment
  + member name
  + member bit/byte offset
  + signedness / integer encoding
  + array length and element schema
  + enum representation
  + nested structural fingerprints
        -> canonical schema digest
```

Normalization 要忽略 BTF-local numeric ID 和不会改变 representation 的 build artifact。得到的 digest 可以写进 application manifest 或 metadata sidecar，在 pinned map reuse 前比较。

真正难的是定义“compatible”而不只是“identical”。只改 typedef 名称不应该强制 migration；显式 padding 变化可能没有行为影响；只读 prefix 的 consumer 也许可以接受 append-only extension。反过来，如果简单忽略 field name，又可能把两个同宽语义字段的交换当成兼容。

因此 artifact 至少要支持两类 policy：严格 structural identity，以及开发者显式声明规则的 compatible evolution。

**Artifact。** 一个 libbpf-adjacent `map-schema` 工具：从 ELF object 和已有 map 提取 canonical BTF fingerprint，输出 human-readable structural diff，并给 loader 一个 machine-readable compatibility result。

**Evaluation。** 建立包含 nested type、array、enum、padding、CO-RE-friendly source change、不同 compiler 和 BTF dedup 变化的 mutation corpus。测 false accept、false reject、普通 rebuild 下 digest stability，以及 startup latency。

**生产价值。** Loader 可以明确报出“reuse 被拒绝，因为 `flow_state.last_seen_ns` 从 offset 0 移到了 16”，而不是上线后才发现状态损坏。

**失败条件。** 如果 canonicalization 在普通 toolchain rebuild 间都不稳定，或者大量 production object 根本没有足够 BTF，这个 fingerprint 就不适合做 mandatory fleet gate；需要 source-generated schema manifest 作为 fallback。

## 方向二：给 map state 显式版本，并在 semantics 改变时做受控 migration

**Gap。** Structural identity 无法表达单位、identifier namespace、validity epoch 或算法 invariant 的变化。

**Mechanism。** 给每个 state-bearing map 一个独立于 bpffs path 的 logical schema contract：

```text
map_identity: flow_state
structural_fingerprint: sha256:...
semantic_revision: 4
lifecycle: must-preserve
migration_from: [2, 3]
reset_policy: forbidden
```

启动时，loader 把已有 map 分成四种结果：

```text
direct reuse
compatible read / migrate before write
explicit reset
hard refusal
```

需要 migration 时，不要直接在旧 representation 上原地修改。创建新 generation map，quiesce 或 generation-gate writer，转换 entry，验证 entry count 和 domain invariants，再把 program/controller 切到新 generation。新版本真正通过后才 retire old map，并保留足够长时间支持 rollback。

这里不是重新发明一套完整 application transaction system。新的部分是：**由 schema evidence 决定哪些 map 可以 direct reuse，哪些 map 必须进入 migration path。**

**Artifact。** Versioned map manifest + migration runner；简单 structural transformation 可以自动生成 adapter，semantic transformation 则调用 application callback。

**Evaluation。** 在持续写入 map 的 rolling controller upgrade 下测试，并在每个 migration phase 注入 crash；加入 LRU eviction、per-CPU value 和“无法同时在内存里保留两份完整 map”的压力场景。测 lost/duplicate update、rollback success、downtime，以及多少 migration 可以安全自动生成。

**生产价值。** 运维能区分“旧 state 有意兼容”和“loader 恰好接受了旧 fd”。对 security、billing、policy 等状态，也可以明确禁止“不确定就 reset”。

**失败条件。** 如果大多数应用把 pinned state 只当 disposable cache，而且重建成本很低，migration machinery 可能比 reset 更贵。lifecycle contract 应该允许 `reset-safe`，而不是强迫所有 map 都持久化。

## 方向三：在给新版本写权限前，用旧状态做 shadow validation

**Gap。** Manifest 本身也可能写错。N 和 N+1 可以错误地声明同一个 semantic revision；migration callback 也可能输出“结构合法但语义不合法”的值。

**Mechanism。** 在 N+1 获得 write authority 前增加 shadow admission phase。让新版本读取 bounded sample 或 snapshot，并检查 declared invariants：

- counter 的范围与 monotonicity；
- identifier 能否在 control plane 正确 resolve；
- 合理的 cross-field invariant，例如适用场景里的 `packets <= bytes`；
- generation / epoch membership；
- old decoder 与 new decoder 对同一 raw entry 的 logical result 是否一致；
- 由 map 推导出的 policy decision 或 metrics aggregate 是否保持 parity。

对 read 无 side effect 的 map，可以让 N/N+1 同时 decode 相同 raw bytes，再比较 normalized logical record。对活跃 map，comparison 必须绑定 epoch 或 snapshot boundary，避免把 concurrent update 错判成 decoder disagreement。

Shadow validation 通过后，loader 生成 reuse receipt，绑定 existing map identity、structural fingerprint、semantic revision、新 object build identity 和 validation result。失败则进入 migration；只有 policy 明确允许时才能 reset，否则拒绝 upgrade。

**Artifact。** 一个 state-compatibility harness，提供 map-type-specific sampling/snapshot adapter 和 invariant plugin。

**Evaluation。** 注入 same-layout semantic bug、错误 schema declaration、corrupted entry、partial migration 和 concurrent update。比较 manifest-only admission 与 shadow validation 能抓到多少 silent bad reuse，测 false alarm、startup delay 和大型 map 下的 coverage。

**生产价值。** Reuse decision 不再只依赖部署前写好的 metadata，而是有一部分真实旧状态上的 behavioral evidence。

**失败条件。** 如果应用没有便宜、无 side effect 的 decoder，也没有足够强的 invariant，shadow validation 很容易变成昂贵形式主义。这种情况下它更适合作为 `must-preserve` 状态的 risk-based option，而不是所有 map 的强制步骤。

## 今天就可以采用的 production guidance

不需要等新 kernel API，loader 现在就能把 reuse 做得更安全：

1. **给所有 pinned map 做 lifecycle 分类。** 标记 `ephemeral`、`reset-safe`、`must-preserve` 或 `migrate`，不要从 bpffs path 猜生命周期。
2. **先做 kernel-visible definition gate。** Map type、key/value size、`max_entries`、flags、`map_extra` 和 map-type-specific constraint 仍然是第一层。
3. **把 schema metadata 跟 artifact 一起发布。** 有 BTF 时优先计算 structural fingerprint，再加明确的 application semantic revision。
4. **不要把 `bpf_map__reuse_fd()` 当 compatibility verdict。** 它应该是 compatibility 已经成立后执行 reuse 的 mechanism。
5. **对模糊的 `must-preserve` state 直接拒绝。** Security、billing、policy、accounting map 不应该在 schema 不确定时 silent reset 或 blind reuse。
6. **Migration 用新 generation。** 验证和 cutover 成功前保留旧 map，开始前就定义 rollback。
7. **记录 evidence。** 日志里留下 expected/observed map parameters、structural digest、semantic revision、migration decision 和最终 map identity，事后才能回答“为什么当时认为这个 state 可以继续用”。

这样可以把 program compatibility 与 state compatibility 清楚拆开：

```text
artifact + target kernel
        -> interface/capability admission

artifact + existing map state
        -> representation/semantic admission

两者都通过
        -> load, attach, grant state authority
```

第一个 path 成功，绝不能自动推出第二个也成功。

## 什么证据会推翻这个结论？

如果真实 production 数据表明，长期 pinned map 几乎总是两类：要么 ABI 多年完全固定，要么只是 disposable cache、每次 application upgrade 都安全 reset，那么 parameter equality + explicit reset 也许已经够用，复杂 schema contract 的收益会很小。

如果 BTF-derived schema 在普通 compiler/toolchain rebuild 间都无法稳定，或者 production deployment 经常剥掉 key/value type information，structural fingerprint 这条路也会变弱；那时 source-generated manifest 可能更实际。

如果大型真实 map 没有一致、低成本的 observation point，而且应用 invariant 太弱，shadow validation 也可能无法区分正确 decode 和错误 decode。

还有一个明确 counterexample：某个 map value 本来就被两个 generation 都当成 opaque fixed-size byte string，从来没有代码依赖它内部的 source-language layout，那么某个 wrapper struct 的“结构变化”就没有行为意义。Compatibility 应该由真实 consumer 定义，而不是教条地要求每个 type spelling 都相同。

最后，如果升级明确要求 reset state，而且这个 reset 符合应用 policy，就根本没有必要保存旧 semantics。Compatibility gate 只在旧状态真的要被 carry forward 时有意义。

当前 Linux/libbpf 的边界仍然留下一个清楚 gap：kernel-visible map parameters 可以证明两个版本能共享同一种 storage shape，BTF 可以暴露很大一部分 representation；但它们都不能单独证明新 application generation 会给旧 bytes 赋予同一个 meaning。更安全的 reuse decision 应该组合 **map-definition compatibility、稳定的 structural schema evidence，以及显式 application semantic contract**；三者不一致时进入 migration 或直接拒绝。

## 参考资料

- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF Type Format (BTF)](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [eBPF Docs: `bpf_map__reuse_fd`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_map__reuse_fd/)
- [eBPF Docs: `BPF_MAP_CREATE`](https://docs.ebpf.io/linux/syscall/BPF_MAP_CREATE/)
- [Eunomia Daily Report: 有状态 eBPF 应用能否原子升级？](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)
- [Eunomia Daily Report: eBPF 加载器能相信内核版本号吗？](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)
- [Eunomia Daily Report: eBPF Object 在 Kernel 升级后还能保持原来的含义吗？](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)
- [Eunomia Daily Report: eBPF 加载器能把 kfunc 简化成“有”或“没有”吗？](https://eunomia.dev/zh/research/ebpf-kernel-interface-negotiation/)
