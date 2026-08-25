---
date: 2026-08-25
title: "eBPF 能验证有状态安全策略，而不只是验证字节码安全吗？"
description: "有状态 eBPF 安全策略依赖跨事件持久化的 map 状态。本文讨论策略状态契约、运行时 transition guard 与 temporal verification。"
tags:
  - Daily Report
  - eBPF
  - Security
  - Verification
  - BPF Maps
research_question: "怎样验证多个 eBPF 程序、CPU 与用户态共同维护的策略状态满足时间上的安全不变量，同时避免把完整策略塞进昂贵的通用 verifier？"
source_cutoff: 2026-08-25
status: daily-report
---

# eBPF 能验证有状态安全策略，而不只是验证字节码安全吗？

网络策略可以在连接建立时做一次判断，把结果记下来，然后允许之后的 reply packet。syscall 策略可以规定“初始化完成后才允许某个操作”。认证缓存也会把先前的一次身份检查变成后续 fast path 的依据。这些安全决定都有一个共同点：**当前这次执行是否允许，取决于之前发生过什么。**

Linux eBPF 很适合实现这类机制。程序可以挂在大量安全相关 hook 上，[BPF map](https://docs.kernel.org/bpf/maps.html) 又能在多次执行之间保存状态，并和用户态共享。Cilium 的 stateful network policy 就会依赖 connection tracking、authentication 和 endpoint policy map。

<!-- more -->

但 Linux [eBPF verifier](https://docs.kernel.org/bpf/verifier.html) 证明的是另一类性质。它沿着程序路径做抽象执行，跟踪 register、stack、pointer 与 scalar 的状态，确保加载进内核的 bytecode 不会越界访问、解引用无效指针或破坏内核执行安全。这套保证非常重要，却不等于“map 里编码的安全状态机一定符合策略意图”。

假设一个简化的认证策略有三部分：某个 hook 写入 `AUTHENTICATED`，另一个 hook 只要看到这条状态就放行，用户态 controller 可以撤销身份，同时 map 在容量不足时可能插入失败或发生 eviction。每一段 BPF 程序都可以通过 verifier，也都可以做到 memory-safe，但组合起来仍可能出错：revocation 后旧状态没有及时失效，两个 CPU 竞争产生非法 transition，旧 controller 写入不属于当前 policy generation 的状态，或者 map pressure 改变了系统到底 fail-open 还是 fail-closed。

真实系统已经说明这种区别不是纯理论问题。Cilium 文档明确把 session-based policy enforcement 定义为 stateful，同时列出 connection-tracking、authentication 和 endpoint policy map 的容量上限。它还支持在 policy map overflow 时把 endpoint 进入 lockdown。这些行为都不是“bytecode 能否安全执行”的问题，而是 **持久状态、transition authority、资源上限与恢复语义** 的问题。

本文讨论的不是把 Linux verifier 扩展成一个能够证明任意分布式状态机的超级 model checker。更实际的方向是：让策略作者声明一个很小的 temporal contract，静态能证明的部分在加载前完成，只有真正依赖运行时状态的条件才使用便宜的 runtime guard。

这和之前的 [stateful eBPF transactional upgrade](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/) 不一样。那篇文章关注的是程序、map、link 与 pinned state 如何跨 generation 原子升级。它也不同于 [multi-tenant network policy composition](https://eunomia.dev/zh/research/ebpf-network-policy-composition/)，后者问的是多个 policy owner 的规则最后如何组合、谁对一个 verdict 负责。这里假设 policy generation 已经正确安装，authority 也清楚，缺的是：**这套策略运行过程中产生的状态 transition 是否一直合法。**

## Linux verifier 主要证明一次程序执行内部的安全性

verifier 对 BPF instruction 做 abstract interpretation。它会跟踪寄存器和 stack slot 的可能值、pointer type 与 branch state。如果 `bpf_map_lookup_elem()` 可能返回空指针，程序就必须先做检查，后续才能安全访问 map value。

这个模型有意把 map 当成一个“可以安全访问的内核对象”，而不是把 map 当前保存的业务含义理解成 temporal specification。generic hash map 里的一段 value 可能是 counter，也可能是 ACL、connection state 或 auth token；内核 verifier 不应该仅凭字节布局去猜策略语义。

这种边界是 eBPF 能保持实用性的原因之一。加载一个程序不需要 model-check 未来所有 packet、syscall、timer callback、用户态 update 与 CPU interleaving。程序一旦通过 verifier，fast path 也不需要每次进入一个通用安全解释器。

但是安全系统会在 map value 上再建立一层语义。例如：

```text
{ subject = 42, state = AUTHENTICATED, generation = 17 }
```

对 map 来说它只是安全可访问的 bytes；对策略来说，它表示 subject 42 在 generation 17 下已经通过认证。map interface 可以保证 memory safety，却不会自动保证这个含义不会过期、越权或被错误 transition。

## Stateful policy 已经是生产需求，不是未来设想

Cilium 是一个很直接的例子。它的 policy 文档说明，对于 session protocol，允许 `A => B` 建立连接，也会自动允许 B 返回 reply packet，但不会因此允许 B 主动建立新连接到 A。这需要 connection state。

Cilium 的 eBPF map 文档还给出了 connection tracking、authentication 与 policy map 的容量。对普通 cache 来说，eviction 也许只意味着命中率下降；对 security state 来说，丢掉一条记录可能改变后续 verdict。因此 Cilium 暴露 map pressure metric，并且可以在 endpoint policy map 无法容纳全部规则时进入 lockdown。

这不是在说明 Cilium 做错了什么，恰恰相反，它说明生产级 eBPF security datapath 的 correctness 本来就依赖 verifier 之外的状态条件：map capacity、endpoint regeneration、identity change、connection lifetime 和 userspace control plane update。

syscall filtering 从另一个方向给出证据。[Programmable System Call Security with eBPF](https://arxiv.org/abs/2302.10366) 指出 classic seccomp/cBPF 很难表达 stateful policy，因此设计了 Seccomp-eBPF program type，并加入安全的 filter state、同步机制以及受控的 kernel/user state access。论文里的 temporal specialization 最多把暴露的 syscall attack surface 降低 55.4%。也就是说，有状态策略本身确实有价值，值得新增执行接口。

但“能表达 state”与“能证明 state machine 是对的”仍然是两件事。

## 相关工作更像是在提示一个 verifier/runtime 分工

近年的系统提供了几块很有用的拼图。

[VEP](https://www.usenix.org/conference/nsdi25/presentation/wu-xiwei) 用 annotation 和 two-stage verification 验证更完整的 eBPF-C 程序，并用较小的 bytecode proof checker 检查结果。它覆盖 map-owned memory 等安全性质，主要目标仍是证明一段程序具有完整而安全的 programmability，而不是证明无限多个 invocation 与 control-plane update 组成的 temporal policy。

[ePass](https://ebpf.foundation/research-update-verifier-cooperative-runtime-enforcement-for-ebpf/) 提出了另一种很适合这里的思路：静态 verifier 无法高效证明的动态性质，可以通过 transformation 插入 targeted runtime checks，同时 verifier 继续作为最终 gatekeeper。stateful policy 正好有这种结构：一部分事实加载时已知，另一部分只有事件真正发生时才知道。

在 eBPF 之外，[p4tv](https://www.usenix.org/conference/nsdi25/presentation/zhang-delong) 已经证明 stateful data plane 可以按照 temporal trace property 来验证，而不只是验证单个 packet 处理函数。但 eBPF 有额外难点：map 是通用数据结构，程序挂在完全不同的 hook 上，同一份 state 可能被多个 BPF program 和 userspace process 修改，而且不同 CPU 会并发执行。

[BPF-DB](https://www.pdl.cmu.edu/PDL-FTP/Database/butrovich-sigmod2025_abs.shtml) 则从数据管理角度补上另一个缺口：普通 BPF map 操作缺少跨多条更新的 transaction guarantee。transaction 可以让一组 state update 原子提交，但它仍然不能回答 `UNAUTHENTICATED -> ADMIN` 这个 transition 到底是不是策略允许的。

这些工作放在一起，更值得研究的并不是“再造一个更大的 verifier”，而是：**能否定义一个很小的 temporal policy contract，让静态部分进入 proof，动态部分通过 bounded transition API 执行。**

## 现有研究还缺什么

### 1. Map type 描述存储行为，却不描述策略允许哪些 transition

BPF map 会定义 key/value、capacity、lookup/update 方式，以及部分 map 的 concurrency 或 eviction 行为。security system 再自行把这些 value 解释成 connection、authentication、policy generation 或 revocation state。

缺的是 machine-checkable transition declaration，例如：

```text
state AUTHENTICATED {
  enter: only auth_hook or trusted_control_plane
  requires: identity_generation == policy_generation
  leave: timeout, revoke, endpoint_delete
}

invariant:
  allow_sensitive_operation => state == AUTHENTICATED
```

最直接的测试方法，是故意从 BPF 和用户态写入 memory-safe 但 policy-illegal 的状态，观察系统能不能在产生 allow verdict 之前拒绝它。

### 2. 独立 hook invocation 之间没有天然的验证边界

verifier 看到的是一段程序路径，但一个 security policy 可能跨 LSM hook、cgroup networking hook、timer callback、另一个 CPU 和 userspace controller。

缺的是对 persistent state 定义一个 bounded transition relation，同时说明哪类 hook/program 有权做哪种 transition，哪些字段必须和 policy generation 一致。它不能要求 kernel 保存或枚举无限长 history。

评测应该包含 race、event reordering、delayed revocation、program restart 和 multi-hook shared state。如果普通 per-program verification 加 unit test 已经能稳定找到同样的问题，那么就没有必要再增加 temporal layer。

### 3. Capacity 与 eviction 对安全状态来说也是语义事件

generic map 都有有限 `max_entries`，LRU map 还可能自动淘汰 entry。对 cache，丢 entry 可能只是变慢；对 security state，丢 entry 可能改变之后的 allow/deny。

因此每类安全状态还缺一个明确的 failure semantic，例如 `fail_closed`、`recompute`、`unknown` 或 `safe_to_evict`。Cilium 的 policy-map lockdown 是一种 production answer，但这种含义并不是 generic eBPF state schema 的一部分。

真正应该测试的是 map 被压到 capacity 时策略 invariant 是否仍然成立，而不是只看 BPF program 有没有继续运行。

### 4. Userspace 也是 transition authority 的一部分

BPF map 天生允许和 userspace 共享，这让动态 control plane 很方便，也意味着 proof 不能假设只有 verifier 检查过的 BPF code 能写状态。

现在缺的是把 configuration write、trusted transition、observation-only access、migration 与 repair 区分开。一个 raw map FD 往往给 controller 比策略真正需要的更大权限。

应当模拟 stale controller、crash/restart 与 concurrent policy generation。如果旧 controller 仍然能制造一个 fast path 接受的安全状态，transition authority 就是不完整的。

## 兼具学术价值与生产价值的方向

### 1. 把 temporal policy contract 编译成 map 与 hook obligation

**Gap.** 当前 BPF type/verifier 能证明 memory 和 pointer safety，却不知道 map field 的策略含义。

**Mechanism.** 定义一个刻意受限的 state schema：state name、allowed transition、authorized hook/program class、generation relation、expiry behavior，以及 capacity failure semantic。compiler 把它生成普通 BPF map layout 与每个 program 的 proof obligation。程序可以自由读取 state，但修改安全关键字段时必须走生成的 transition wrapper。

目标不是验证任意 C，而是把少量真正影响 policy invariant 的 mutation 收敛进一个可检查接口。其他 BPF logic 继续使用现有 C/Rust 和 Linux verifier。

**Delta.** VEP 验证 annotated eBPF program 的安全；p4tv 在 P4 特定数据面里验证 temporal behavior。这里保留 Linux verifier 负责通用 bytecode，只新增一个很窄的 cross-invocation policy-state language。

**Artifact.** 一个 schema compiler、libbpf integration、生成的 map value/transition wrapper，以及加载时消费的 compact manifest。

**Evaluation.** 实现 network、syscall 与 BPF-LSM 三类 stateful policy，注入 illegal transition、stale generation、missing expiry 与 update ordering bug。比较 handwritten BPF、property-based tests、适用场景下的 VEP-style per-program verification 和 contract compiler。指标包括 violation detection、false reject、build/verifier time、code size 与 runtime overhead。

**Academic value.** 核心问题是，在 temporal verification 变得不可计算之前，能把多少安全语义压缩成一个小而 decidable 的 contract。

**Production value.** security team 可以直接 review 一张 state machine，而不需要从分散的 map write 与 controller code 反推策略。

**Failure condition.** 如果真实 policy 经常必须绕过 wrapper 直接写 map，说明这个抽象过弱。

### 2. 对动态 transition 使用 verifier-cooperative runtime guard

**Gap.** 当前 policy generation、entry 是否被 eviction、哪一个 controller 持有 repair lease、revocation 是否刚好 race，这些事实在 load time 无法知道。

**Mechanism.** 借鉴 ePass 的 verifier-cooperative 思路，把 security-critical update 转换成一个很小的 guarded transition API。static verifier 负责证明参数、pointer ownership 和 bounded execution；runtime guard 只检查 `old_state`、generation、transition authority、expiry epoch 等动态 predicate。

一种实现可以是带 typed state descriptor 的 kfunc 或 map wrapper。common fast-path transition 只做少量比较和一次 update；复杂 recovery 留给 userspace。非法 transition 返回明确 reason，并按照 schema 选择 fail-closed 或 recompute。

**Delta.** BPF-DB 提供通用 transaction，ePass 提供和 verifier 配合的 runtime safety check。这里的 runtime path 更窄，只执行 policy semantic transition，不试图成为通用 DB 或通用 bytecode sandbox。

**Artifact.** prototype transition-map/kfunc API、表示 transition authority 的 verifier annotation，以及把 state descriptor 绑定到 program/map bundle 的 libbpf tooling。

**Evaluation.** 测 packet/syscall throughput、P50/P99 latency、cache miss、map memory 与 failed-transition cost；压力场景包括多 CPU concurrency、map pressure、revocation storm、controller crash/restart 与 rolling update。通过 ablation 区分 static check 和 runtime guard 各自能抓到哪类 bug。

**Academic value.** 这是一个清晰的 hybrid verification boundary：哪些 security invariant 适合静态证明，哪些可以用很便宜的 runtime check 补齐。

**Production value.** operator 可以在内核 fast path 直接 fail closed，而不是把每次事件都送回 userspace policy engine。

**Failure condition.** 如果 guard 的 cache/synchronization 开销明显破坏 fast path，或者实现复杂度和原 policy 一样高，就应该继续使用专门实现和测试。

### 3. 建一个专门制造 state fault 的 temporal eBPF policy benchmark

**Gap.** verifier test suite 主要验证 bytecode safety 与 verifier 本身。security policy 测试通常只检查预期 packet/syscall，很少有统一 ground truth 来覆盖 concurrency、capacity pressure 与 control-plane failure 下的 state-machine violation。

**Mechanism.** 定义一些小 policy automata，然后生成合法 trace 和 adversarial perturbation：duplicate event、reordering、concurrent writer、stale generation、LRU eviction、full map、userspace crash、BPF replacement 与 delayed revocation。每条 trace 都标注最终 allow/deny/unknown 和最早违反 invariant 的 transition。

可以把 P4 temporal verification 作为一个 baseline，但 benchmark 必须增加 eBPF 特有维度：heterogeneous hook、userspace map writer、per-CPU state、不同 map semantic 与 program generation。

**Artifact.** open corpus、fault injector、replay harness，以及 kernel eBPF security system adapter。

**Evaluation.** 主指标应该是 false allow、false deny、漏掉的 invalid transition、定位第一个错误 transition 的时间，以及固定 state budget 下的 throughput/overhead。capacity/recovery bug 要和普通 policy logic 分开报告。

**Academic value.** 有共同 benchmark 后，stateful policy correctness 才能从“program load 成功”变成可比较的 temporal property。

**Production value.** CNI、runtime security 与 syscall-policy 项目可以把很多只在压力和恢复阶段出现的问题变成 regression test。

**Failure condition.** 如果真实事故都依赖无法抽象的 application-specific semantic，那么这个 corpus 应该保持为多套 system-specific tests，而不是宣称存在统一 benchmark。

## Verifier 更合理的边界是什么

Linux verifier 最适合继续做它已经很擅长的事情：在 BPF execution model 下证明程序能安全进入内核执行。让它理解每一种 security policy 的全部 history，不仅会放大 state explosion，也会让 verifier 对无法自动推断的 application semantics 负责。

更实际的边界是把 **security-critical persistent state 显式化**。program 声明一个小 transition contract；static tooling 证明能静态确定的部分；verifier 保证程序只能通过批准接口修改关键状态；真正依赖实时状态的 predicate 由 runtime guard 检查；长 trace 和 recovery 再由 temporal model checking 与 fault injection 覆盖。

这样也会改善诊断。事故发生时，问题不再是“BPF program 都过 verifier 了，为什么还会放错包？”而可以精确到：哪个 transition 创建了当前 state，它属于哪一代 policy，哪条 invariant 授权了这次 allow。

## 哪些结果会改变这个判断？

最强的反对意见很现实：现在大量 stateful eBPF policy 已经靠普通 map、仔细设计的 control plane、unit test、fuzzing 和 production monitoring 运行得很好。增加 temporal contract 可能只是多了一门语言、一个 checker 和一次 fast-path check。

以下结果会明显削弱本文的方案：

1. 一组有代表性的 eBPF security system 显示，安全相关 transition 太依赖具体应用，连一个很小的公共 schema 都无法表达；
2. property-based testing 与 fault injection 用更低成本就能找到几乎所有 transition bug；
3. runtime guard 带来的 cache traffic 或 synchronization 足以抵消 in-kernel enforcement 的性能优势。

相反，如果真实故障不断来自 stale generation、concurrent map transition、capacity pressure，或者 control plane 与 BPF datapath 对 state 的理解不一致，那么这个方向会更有价值。

因此下一步最有说服力的证据，不是再做一个 verifier microbenchmark，而是找到一批 **memory-safe、verifier-safe，但因为 persistent state 走了非法 trace 而做出错误安全决定的 eBPF policy**。

## References

- [Linux kernel: eBPF verifier](https://docs.kernel.org/bpf/verifier.html)
- [Linux kernel: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Cilium: Policy Enforcement](https://docs.cilium.io/en/latest/security/network/policyenforcement/)
- [Cilium: eBPF Maps](https://docs.cilium.io/en/latest/network/ebpf/maps/)
- [Cilium: Endpoint Lifecycle and policy-map lockdown](https://docs.cilium.io/en/stable/security/policy/lifecycle/)
- [Programmable System Call Security with eBPF](https://arxiv.org/abs/2302.10366)
- [VEP: A Two-stage Verification Toolchain for Full eBPF Programmability](https://www.usenix.org/conference/nsdi25/presentation/wu-xiwei)
- [On Temporal Verification of Stateful P4 Programs](https://www.usenix.org/conference/nsdi25/presentation/zhang-delong)
- [BPF-DB: A Kernel-Embedded Transactional Database Management System for eBPF Applications](https://www.pdl.cmu.edu/PDL-FTP/Database/butrovich-sigmod2025_abs.shtml)
- [ePass: Verifier-Cooperative Runtime Enforcement for eBPF](https://ebpf.foundation/research-update-verifier-cooperative-runtime-enforcement-for-ebpf/)
