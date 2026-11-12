---
date: 2026-09-10
title: "同一个 eBPF 操作放到 CPU、NIC 和 DPU 上，语义还能一致吗？"
description: "同一个 eBPF 逻辑操作可以交给 CPU、NIC 或 DPU 实现。本文分析跨 backend 时怎样保持原子性、顺序、失败语义和状态转换一致。"
tags:
  - Daily Report
  - eBPF
  - SmartNIC
  - DPU
  - Offload
  - Verification
research_question: "eBPF 怎样让一个高层语义操作同时拥有 CPU、kernel、NIC 和 DPU 实现，又保持相同的状态转换、并发、顺序与失败语义？"
source_cutoff: 2026-09-10
status: daily-report
---

# 同一个 eBPF 操作放到 CPU、NIC 和 DPU 上，语义还能一致吗？

假设一个 XDP 应用维护 per-flow state，并暴露一个逻辑操作：更新 flow record，然后返回新的决策。在一台机器上，它操作普通的 kernel BPF hash map；在另一台机器上，program 和 map 被 offload 到 NIC；在第三种部署里，同一个操作由 DPU runtime 对 device-local state 执行，再周期性和 host 同步。

只做最简单的测试，这三个实现可能完全一样。给一个 key、一个 value，不制造并发，每个 backend 都返回 success，而且最后读出的 record 也一致。但这还不能证明它们实现的是同一个操作。

真正麻烦的是两个 packet 同时更新同一个 flow、device queue 改变执行顺序、一次 update 在部分生效后失败、host fallback 与 device 上还没结束的执行发生竞争，或者 state 正在 backend 之间复制的时候。Linux 文档明确说普通 BPF hash map 的 `bpf_map_update_elem()` 会原子替换已有元素；而 Linux 的 BPF offload 路径会把 lookup、update、delete 和 iteration 交给 device-specific map operation。只要一个高层操作开始拥有多个执行实现，**语义等价就不能只比较一次调用返回了什么，还必须包括 state transition、atomicity、visibility、ordering 和 failure behavior。**

<!-- more -->

本文继续前面的 [跨架构 eBPF 特化](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/) 与 [native-operation trust boundary](https://eunomia.dev/zh/research/ebpf-native-operation-trust-boundary/) 往下走。前面的文章分别讨论 implementation 是否适合当前机器、是否存在 portable fallback，以及 native implementation 到底需要额外信任什么。这里把这些问题都假设为已经解决：两个 backend 都合法、身份也已知，operator 也愿意信任它们。剩下的问题是，真实并发和故障出现以后，它们到底有没有实现**同一个有状态操作**。

它也不同于之前的 [host/offload complete mediation](https://eunomia.dev/zh/research/ebpf-complete-mediation-offload/)。complete mediation 关心每一条 policy-relevant packet path 是否仍然经过当前 generation 的 enforcement point。本文假设该经过的路径都经过了，只问 enforcement point 里面执行的那个 operation，在不同 backend 上是不是具有同一套语义。

## BPF 接口本身就包含比返回值更丰富的语义

BPF map 很适合作为起点，因为它表面上只是一个简单 key/value API，但不同 map type 实际上已经携带不同的状态语义。

Linux 对 `BPF_MAP_TYPE_HASH` 的文档说明，`bpf_map_update_elem()` 会原子替换已有元素。per-CPU hash map 则故意采用另一种 state model，每个 CPU 有独立 value slot。LRU 版本还带 eviction semantics；如果 map value 里有 spin lock，又需要明确的 locked access。也就是说，同一个“map operation”背后已经包含 visibility、ownership、eviction 与 synchronization，而不只是一个 return code。

UAPI 的 failure semantics 也能被应用观察到。`BPF_MAP_UPDATE_ELEM` 通过 `BPF_ANY`、`BPF_NOEXIST` 和 `BPF_EXIST` 区分 create-or-update、create-only 与 update-only，并可能返回 `EEXIST`、`ENOENT` 或容量相关错误。batch operation 在出错时甚至可能报告只有前面一部分 element 已经处理成功。应用完全可能依赖这些细节，即使源码里看起来只是“update 一下 map”。

所以对于 delegated operation，只在 happy path 上得到同样的最终 value 远远不够。两个 backend 仍然可能对“哪些并发 history 合法”或“一个 failure 到底意味着什么”给出不同答案。

## Linux offload 已经把 BPF-facing operation 路由给 device implementation

当前 Linux 的 `kernel/bpf/offload.c` 把这条实现边界写得很清楚。对于 offloaded map，kernel 里的 `bpf_map_offload_lookup_elem()`、`bpf_map_offload_update_elem()`、`bpf_map_offload_delete_elem()` 与 `bpf_map_offload_get_next_key()` 会进一步调用 device-specific `dev_ops`。Program offload 也会通过 device callback 完成 verifier preparation、instruction hook、finalization、translation 和 teardown。

这是很干净的 extensibility boundary。host 仍然保留 BPF-facing object 与 lifecycle，真正的 implementation 则可以在 device 侧。但这个 dispatch interface 本身并不等于一份 formal specification，无法自动证明每个 device implementation 的 concurrency 与 failure semantics 都和 host map type 或另一个 accelerator implementation 相同。

当 operation 是 stateless 时，这个差距比较容易缩小。比如 rotate 或 bit select 可以直接定义成 input register 到 output register 的纯函数。Kops 正是利用了这种形状：一个 verifier-visible BPF proof sequence 配一个 native emit，EInsn operation 再用 Lean 4 证明两边计算结果等价。

一旦 operation 有状态，observable boundary 就大得多。两个 implementation 可以算出同一个局部结果，却在 linearization point、另一个 worker 何时可见、error 后是否 rollback、backend transition 中的 in-flight update 怎么处理等方面完全不同。

## ISA portability 不能直接推出 semantic-operation portability

[RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) 为 BPF 提供 platform-neutral instruction-set specification 与 conformance group。这个层次很适合定义“一条 BPF 指令是什么意思”以及一个 runtime 支持哪些 instruction group。

高层 operation 面临的却是另一类问题。假设一个 `flow_update_v1` 同时存在四个 implementation：

```text
host_hash_map     -> 按 Linux hash-map semantics 更新
host_native_fast  -> kernel/native optimized implementation
nic_table         -> 更新 device-local flow table
DPU_service       -> 通过 RPC 或 shared memory 更新 DPU-owned state
```

Capability negotiation 可以告诉 runtime 四个 implementation 都“能跑”。proof 或 code review 可以说明每个 implementation 单独看是安全的。execution provenance 又可以告诉 operator 事故时到底跑了哪一个。可是这些信息都没有定义：并发 invocation 是否 linearizable，success 是否在 reset 后仍然成立，timeout 会不会表示“其实已经 commit 只是 reply 丢了”，或者 host fallback 是否可能看到还没从 device 变得 visible 的状态。

这些都属于 operation semantics。如果不显式写出来，“same operation”最后只剩下同一个 API name，而不是一个 correctness statement。

## 有状态等价需要 observable transition contract

一个真正有状态的 semantic operation 需要的是对 state 的 reference relation，而不只是 input/output signature。

对一次调用，可以把抽象操作写成：

```text
(result, S') = OP(args, S)
```

一旦有并发，contract 还需要定义哪些 history 是允许的。implementation 可以承诺 linearizability、per-key serialization、eventual visibility，或者只保证同一个 queue 内部有序。这些 contract 都可能合理，但它们不是同一个 contract。问题不在于一定要选最强保证，而在于选择必须显式。

Failure behavior 也应该进入同一个模型。如果 device 返回 `-EIO`，abstract state 是否保证完全没变？update 会不会已经 commit，只是 caller 看到失败？retry 是否 idempotent？backend reset 会不会丢掉已经返回 success 的 operation？如果 implementation 回答不了这些问题，runtime 就不能因为它和另一个 backend 函数签名相同，就安全地在两者之间切换。

这里需要区分 *implementation capability* 与 *operation semantics*。前者只说明“这个 backend 能执行”，后者才说明“哪些 observable history 可以算正确”。

## 现有研究还缺什么

第一个缺口是 **state-transition equivalence**。现有 verifier 很擅长证明 BPF execution 的安全性质，native-operation 工作也可以对有限 instruction sequence 做 equivalence proof，但对于“两个异构实现是否 refine 同一个 state machine”，目前缺少同样成熟的机制，尤其当 state 的保存、同步与失败方式都不同的时候。

第二个缺口是 **跨 backend 的 concurrency semantics**。host BPF map 有 map-type-specific 的 synchronization 与 visibility；device table 会有自己的 queue、atomic primitive、batching 与 memory domain。backend-neutral operation 需要声明 linearization 或 visibility model，否则从 host 移到 NIC 时，合法并发结果集合可能已经变了，即使一条 BPF bytecode 都没有改。

第三个缺口是 **failure equivalence**。当 implementation 可能在部分或不可逆 progress 之后失败，只保持 return-code compatibility 太弱。timeout、reset、queue overflow、DMA failure、firmware restart 与 host-device disconnect 都可能让系统不知道 operation 到底有没有发生。portable semantic operation 应该把这种 uncertainty 暴露出来，而不是把所有 backend-specific failure 都压成一个 generic error。

第四个缺口是 **fallback 或 migration 过程中的 transition correctness**。host implementation 和 device implementation 即使分别正确，混合执行仍可能破坏抽象 contract。某个 request 在 device 上刚完成，fallback path 却从 stale host state 开始；或者 table copy 刚好漏掉一个 in-flight update。这里不是再问“有没有 fallback”，而是问 fallback 产生的联合 history 是否还属于 operation 允许的语义。

## 兼具学术价值与生产价值的方向

### 1. 定义 backend-independent operation transition contract

可以在多个 implementation 上方增加一份很小的 machine-readable semantic contract。它记录 operation version、argument/result type、涉及的 abstract state、atomicity 或 linearization guarantee、visibility domain、ordering constraint、failure class、retry/idempotence rule，以及执行期间预期的 state ownership。

例如：

```text
operation = flow_update_v1
state = flow_table[key]
atomicity = per_key_linearizable
success = new_value_visible_before_return
failure = {no_effect, outcome_unknown}
retry = idempotent_if(request_id_matches)
ownership = one_active_backend_per_generation
```

artifact 可以先是 ELF/BTF sidecar，由 loader 或 userspace runtime 消费，不需要一开始就变成新的 kernel ABI。每个 backend 注册自己实现的 contract version；只有 semantic version 与 guarantee 满足应用要求时，loader 才允许选择它。

评估应该故意放入两个 single-thread result 完全相同、但 race behavior 不同的实现。Ground truth 是 abstract state machine 及其允许 history。测量 semantically weaker backend 被错误接受的比例、contract-check overhead，以及应用代码可以移除多少 backend-specific workaround。Ablation 可以分别删掉 atomicity、failure 或 ownership field，看哪些反例重新变得合法。

学术问题是，面向 heterogeneous BPF operation 的最小 transition language 到底需要表达什么。生产价值则是让 kernel、NIC 与 DPU 混合部署之前的 backend substitution 变得可以 review。

### 2. 给每个 stateful operation 配一个 executable reference model

纯 instruction equivalence 可以直接比 output；stateful operation 则需要 history oracle。可以给每个 operation package 携带一个慢速 reference implementation 或小型 transition model，然后让所有 backend 对着同一个 model 做验证。

顺序执行时 differential testing 可能已经足够。并发时则记录 invocation、completion、request identity、result 与相关 abstract state，再检查 observation 是否能 linearize 或 refine 到 contract。对 failure，应该在 backend 定义的不同 commit point 主动注入故障，并要求 implementation 把最终状态分类成 committed、not committed 或显式 `outcome_unknown`。

artifact 是一套 conformance harness，可以对 host BPF map、native fast path 和 device implementation 运行同一个 operation package。Kops 风格的 proof 仍然适合其中 pure sub-operation；更高层 checker 负责那些很难压成一条 register-equivalence theorem 的 stateful history。

评估报告找到多少 semantic counterexample、test generation cost、checking time，以及 concurrency/failure point coverage。如果普通 unit test 能用显著更低复杂度捕获同一组 divergence，这套 reference model 就不值得引入。

### 3. 做 mixed-backend continuity benchmark，而不是再做一个 throughput benchmark

最难的 bug 通常出现在 transition，所以 benchmark 应该主动制造 transition。

从一个 reference behavior 已知的 stateful XDP workload 开始，分别在 host implementation 与一个 device implementation 上运行。保持 request 并发的同时触发 backend handoff、queue drain、reset、table copy、stale-state injection 与 fallback。给 request 加 ID，让 oracle 可以发现 lost update、duplicate effect、impossible reordering、stale read 和 ambiguous outcome。

至少比较四种设计：host-only、device-only、只复制 state 然后切 pointer 的 naive fallback，以及在切换 active backend 前先建立 state frontier 的 contract-aware handoff。吞吐和延迟仍然要测，但主要 correctness metric 应该是 observed history 中有多少无法被声明的 operation semantics 解释。同时报告 transition pause、state-transfer overhead，以及多少 operation 必须标记为 `outcome_unknown`。

学术价值是把 heterogeneous execution 与可以 formalize 的 state semantics 放进同一个 evaluation target。生产价值则是 firmware、driver 和 runtime 升级的 regression gate：一个新 backend 不能只因为更快、single-call test 通过，就被认为和旧 backend 可互换。

## 哪些结果会改变这个判断？

如果现有 BPF offload 与 accelerator interface 已经提供完整、versioned 的 atomicity、ordering、visibility、failure outcome、retry 与 state continuity contract，而且不同 backend 日常就会针对这份共同 contract 做 conformance test，那么再增加一层 semantic operation abstraction 的价值会明显下降。

如果有价值的 heterogeneous specialization 基本都是 pure operation，问题也会小很多。假如 NIC、DPU 与 native fast path 只替换 stateless instruction sequence，而所有 mutable state 都留在 host，并继续严格服从现有 BPF map semantics，那么 Kops 这种 local equivalence 加普通 capability negotiation 可能已经覆盖大部分需求。

还有一个更直接的反例来自实验。如果大规模 mixed-backend test 表明，只要 return value 和最终 state 相同，就几乎不会出现额外 semantic divergence，那么显式 concurrency/failure contract 可能只是工程负担。本文提出的 benchmark 正好可以用来证伪这个假设。

在这些证据出现之前，一个 higher-level eBPF operation 不能因为 CPU、NIC 和 DPU 都实现了同一个 symbol 就被称为 portable。对于有状态 delegation，真正的 portability 是每一个被接受的 implementation，包括切换 implementation 时产生的 history，都能 refine 到同一份显式 observable state-transition contract。

## References

- Linux kernel documentation, [BPF maps](https://docs.kernel.org/bpf/maps.html), accessed 2026-09-10.
- Linux kernel documentation, [BPF_MAP_TYPE_HASH, with PERCPU and LRU Variants](https://docs.kernel.org/bpf/map_hash.html), accessed 2026-09-10.
- Linux kernel source, [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c), accessed 2026-09-10.
- Linux kernel source, [`tools/include/uapi/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/tools/include/uapi/linux/bpf.h), accessed 2026-09-10.
- IETF, [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), October 2024.
- Yusheng Zheng et al., [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
