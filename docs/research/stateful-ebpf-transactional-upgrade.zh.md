---
date: 2026-08-10
title: "有状态 eBPF 应用能不能原子升级？"
description: "单个 BPF link 可以原子替换程序，但有状态应用还跨越 maps、pins 与用户态控制器。本文分析如何实现可提交、回滚和崩溃恢复的整体升级。"
tags:
  - Daily Report
  - eBPF
  - Runtime Systems
  - State Migration
  - Linux
research_question: "一个有状态 eBPF 应用跨越多个程序、link、map、pinned object 和用户态控制器时，需要什么事务语义，才能避免升级过程中暴露半新半旧的配置？"
source_cutoff: 2026-08-10
status: daily-report
---

# 有状态 eBPF 应用能不能原子升级？

假设一个生产环境里的策略应用由两个 eBPF 程序组成：一个挂在 ingress，另一个挂在 cgroup hook。两边都读取同一个 pinned policy map，用户态控制器持续把外部策略同步进去。现在 version 2 不只改了程序逻辑，还改了 map value 的布局。

单独替换某个程序并不难。Linux 有 `BPF_LINK_UPDATE`，一个 BPF link 可以从旧程序切换到新程序，而不需要先 detach 再 attach。真正麻烦的是应用剩下的部分：如果先迁移 map，version 1 可能读到 version 2 的状态；如果先换程序，version 2 可能读旧 schema；如果第一个 link 更新成功、第二个失败，机器会运行一个从未测试过的混合版本；如果控制器恰好在中间崩溃，连“应该回滚到哪里”都不再显然。

<!-- more -->

这就是**替换一个 BPF program**和**升级一个有状态 eBPF application**之间的边界。Linux 已经提供不少有用的对象级原子操作。libbpf 把 load 和 attach 分开，因此程序开始执行前可以准备 state。bpffs pin 让对象生命周期不依赖某一个控制器进程。map indirection 可以把一个逻辑 map 指向新的 inner map。Cilium 这类生产系统也已经自己实现 regeneration、migration、revert 和 finalization。

因此缺的并不是“再做一个能换程序的 API”。从今天检查的内核接口和生产案例来看，更具体的空白是：**缺少一个通用的应用级 commit protocol，把这些对象级操作绑定成同一个 upgrade generation**。一个有用的事务层应该允许控制器先完整准备下一代，显式声明 state 是复用、迁移还是重建，验证跨对象 invariant，再通过一个逻辑 commit point 激活新版本，同时保留旧 generation 直到可以安全 drain 或 rollback，并且在控制器半路死亡之后能够恢复。

这篇报告继续前两篇 Daily Report：[用户态 eBPF runtime contract](https://eunomia.dev/zh/research/userspace-ebpf-runtime-contract/) 和 [eBPF hook composition contract](https://eunomia.dev/zh/research/ebpf-hook-composition-contract/)。前两篇分别讨论 lifetime、capability 和多程序 composition。stateful upgrade 正好是 lifetime、composition 与 persistent state 汇合的地方。

## Linux 已经能安全完成不少单对象切换

要讨论 transactional upgrade，第一步不是重做 Linux 已经有的东西，而是把现有 primitive 的边界说清楚。

### 一个 BPF link 可以无 detach gap 地替换程序

Linux `bpf()` syscall 文档定义了 `BPF_LINK_UPDATE`：把指定 `link_fd` 关联的 eBPF program 更新为 `new_prog_fd`。在 libbpf 这一层，对应接口允许应用直接替换 link 背后的程序，而不是销毁 link 再重新创建。

这解决了很实际的生命周期问题。tracer 更新一个程序时，不需要故意制造“这一小段时间没有任何程序挂着”的窗口。policy program 也可以保持 attachment object 的 owner 和 lifetime，只改变它执行的代码。

但这个操作只描述**一个 link 和一个 new program**。它没有说两个 links、三个 maps、一组 pinned objects 和 userspace controller 如何一起变化。因此它是 transaction 的 building block，不是 application transaction 本身。

### libbpf 本来就把 load 和 attach 分成两个阶段

Linux 的 libbpf overview 把 BPF application 定义成一个或多个 programs、maps 和 global variables，并把生命周期分成 open、load、attachment、tear down。load 阶段会创建 map、完成 relocation、验证并把 program load 进 kernel，但此时 program 还没有开始执行。文档明确指出，这允许 userspace 在没有 program execution race 的情况下先设置初始 map state。

这已经很接近 transaction 里的 prepare phase。version 2 可以先 open、load、populate，而 version 1 继续服务。真正缺的是：当多个 hook 需要一起变化时，什么动作代表“prepared collection 现在正式成为 active application”。

### pinned object 故意跨越 controller process lifetime

BPF syscall 文档使用 reference-based lifetime。map 和 program 可以在进程之间共享，BPF object 也可以 pin 到 bpffs；只有 file descriptor、pin、attachment 等 reference 都消失之后，对象才会被释放。

这对升级很有用，因为 controller 不需要靠一个进程一直活着来维持所有对象。但它也让 crash recovery 变成真实问题：controller 重启后，旧 generation 和只准备了一半的新 generation 可能都还在。正确的 upgrader 必须判断谁是 active、谁是 prepared、谁可以 GC，而不是假设进程退出以后系统就自动回到干净状态。

### map indirection 可以先准备新 state object 再切引用

Linux 的 map-of-maps 允许 outer map 持有 inner map reference。userspace 可以更新 outer map entry，BPF program 通过 outer map lookup 获得 inner map。这提供了一个很实用的 indirection：新 inner map 可以先离线创建和填充，再让未来的 lookup 转过去。

但这依然只解决一个 map entry 的 reference update，而且 outer map 对 inner map type/layout 有约束，不支持多层 nesting。更重要的是，它不会同时替换 program links。这个 primitive 有价值，恰恰因为它展示了“单点 publish”可能是什么样，也把边界暴露出来了。

## 复用 state 和迁移 state 是两种不同的升级

很多 eBPF 升级其实比开头那个例子简单。如果 version 2 使用完全相同的 map schema，而且 state semantics 没变，最安全的方案可能不是 migration，而是直接复用旧 map。

libbpf-rs 的一个讨论里，libbpf maintainer 把这种做法描述为典型 code-upgrade workflow：先 open 新 object，在 load 之前把新的 map object 绑定到旧 pinned map 的 FD，然后 load 新 programs。这样新代码继续使用旧 state。这个例子很重要，因为它直接否定了“所有 eBPF 升级都需要 transaction layer”这种过宽的结论。对单程序、稳定 state 的情况，map reuse 加一次 link update 往往已经够了。

问题出现在“compatibility 不是结构兼容，而是语义兼容”的时候。

例如 map value 从：

```c
struct policy_v1 {
    __u32 verdict;
    __u32 flags;
};
```

变成：

```c
struct policy_v2 {
    __u32 verdict;
    __u32 flags;
    __u64 epoch;
};
```

loader 可以发现 size mismatch，却无法自动知道 `epoch` 应不应该初始化成 0、从别的 source 计算，还是必须等 controller 完成某个同步。甚至两个 struct byte size 完全一样，也可能语义不兼容，例如某个字段从普通 counter 变成 lease expiration，或者 reset ownership 发生变化。

Cilium 的生产 issue 给出了这种区别的直接证据。一个旧 issue 里，pinned map property 与新程序不兼容时，loader 会删除旧 map 以允许 property upgrade，并明确提示预期会有 data loss。重点不是这种行为“错了”，而是 schema incompatibility 迫使系统做一个生命周期决定，而这个决定不属于 verifier 的普通 memory-safety 问题。

所以 stateful upgrade 至少应该明确区分三种模式：

1. **Reuse**：旧新程序确实共享同一套 state semantics。
2. **Transform**：state 必须经过 versioned conversion 才能切换。
3. **Replace**：旧 state 可以丢弃，或者能从别处独立重建。

把三者都隐藏在“load 新 object”后面，会把决定 correctness 的核心步骤藏起来。

## Partial failure 在生产 BPF control plane 里不是理论问题

做 upgrade transaction 的理由，不是为了 API 看起来更整齐，而是 multi-step datapath regeneration 本身就有能暴露错误状态的 failure path。

Cilium 在 2025 年的一个 issue 记录了 endpoint regeneration retry 之后 policy map 变空，最终导致流量被 policy denied 的问题。这个案例尤其有价值，因为维护者沿着具体阶段定位问题：policy state、BPF collection load、attachment、policy-map sync，以及 deferred revert behavior。一个 maintainer 把期望顺序概括为：创建 policy map，先填好 policy，再把 map 注入 program，load program，最后 attach。另一个 maintainer 则追踪到 early return 如何在后续同步之前触发 revert path，把 map 恢复到了错误的空 state。

这不是“Linux BPF update 普遍不安全”的证据。它说明的是：**真实 BPF application 的 side effect 顺序与 rollback 语义已经超出了单个 syscall**。Cilium 之所以存在 revert stack 和 finalizer，就是因为一次成功 regeneration 本来就不是一个操作。

这也给 novelty gate 一个很好的反例。“给 eBPF loader 加 rollback”太弱，因为生产 loader 早就在做。更值得研究的问题是：这些 bespoke state machine 能不能被压缩成一个 portable generation protocol，并且在不同 application 上验证一致的 property。

## Application-level eBPF transaction 到底应该保证什么？

把 upgrade 叫“atomic”很容易说过头。kernel 不可能撤回 version 1 已经处理完的 packet，两个独立 event 也可能在 commit 的前后分别执行。真正有用的 guarantee 应该更窄。

对一个声明出来的 application generation `G`，upgrader 至少应该定义：

- **Prepared completeness**：`G+1` 需要的 program、map、link target 和 controller dependency 在激活前全部存在并通过 validation。
- **State compatibility**：每个 persistent state object 都明确声明 reuse、transform 或 replace，并带 version 和 migration rule。
- **Single logical commit**：系统有唯一权威的 active-generation decision，不能靠“哪个 link 先更新完”来猜版本是否已经生效。
- **No unsupported mixed generation**：如果 application 声明两个 component 必须一起变化，就不能把一个 old、一个 new 的组合长期暴露给 datapath。
- **Recoverable ownership**：controller crash 以后，durable metadata 足以区分 active、prepared、retiring 和 orphaned objects。
- **Bounded retirement**：old generation 在 rollback window 和 in-flight work 安全结束之前仍然保持 reference。

这些都是 application-level property。Linux 底层仍然负责 reference count、verifier、link lifetime，以及各种 map/hook 自己的 synchronization。transaction layer 的工作应该是组合这些 primitive，而不是假装自己重新实现内核。

## 一个 generation protocol 可以把生命周期变得明确

比较实用的模型是四个 phase：**prepare、migrate、commit、retire**。

在 **prepare** 阶段，controller open/load version 2，但不把它接到 production hook。它创建 versioned state，或者绑定经过验证的 existing map，同时记录 expected old generation 和所有 object identity。

在 **migrate** 阶段，controller 把 state copy、transform 或 reconstruct 到新 generation。如果 old state 仍持续更新，就必须显式选择同步策略：短暂 pause write、snapshot + delta、controller dual-write，或者 old/new 共同使用 compatibility map。不同 workload 可能有不同答案，关键是把答案暴露出来并且测量成本，而不是藏在 loader 里。

在 **commit** 阶段，runtime 改变权威的 active generation。对于可以长期保留 stable dispatcher 的 hook，可以让每个 dispatcher 读取同一个 application-wide generation selector，再跳到该 generation 对应的 program slot。这样多 link cutover 在逻辑上压缩成一个 generation decision，代价是 hot path 多一层 indirection。对于不适合这种 dispatch 的 hook，则必须明确降低 guarantee，或者最终需要更强的 kernel primitive。

在 **retire** 阶段，old generation 继续被 pin 或持有 reference，直到 runtime 能确认不再需要 rollback。stateless hook 可能一个 grace period 就够了；stateful application 则可能需要 controller acknowledgement、异步 work 完成，或者确认没有 state reference 仍指向 old map。

这个 protocol 一开始不需要成为 universal kernel ABI。它首先应该是一个能比较 implementation 的 model：哪个 backend 能提供完整 guarantee，哪个只能提供弱一些的 guarantee，都要说清楚。

## Where current work is still weak

### 现在没有 portable object 描述“一个 upgrade generation 包含什么”

今天检查的接口主要暴露 programs、maps、links、pins 和 object files。生产系统则自己维护 regeneration context 与 rollback state。缺的东西是一个 portable manifest，声明哪些 object 属于同一代 application、它们替代哪些旧 object，以及 commit 前必须满足哪些 compatibility relationship。

没有这个对象时，crash recovery 只能从 pin name、process state、loader convention 或外部数据库里猜 intent。leaked object 只是资源问题，但删错一个“看起来已经旧了”的 map 就会变成 correctness 问题。

最直接的验证方式，是拿几个已有 loader 的 upgrade state machine，看它们是否能用一套很小的 generation states 和 dependency edges 表达。如果每个 loader 都需要完全不同的语义，那 common manifest 就不是对的 abstraction。

### map schema compatibility 目前主要还是结构检查

loader 可以比较 map type、key/value size、flags 等 property，BTF 也能提供丰富 type information。但这些信息本身不能证明 semantic migration 正确。

缺的是把 old schema、new schema、transformation 和 invariant 连接起来的 migration contract。否则 generic loader 最多只能 reuse、reject 或 recreate，复杂 project 继续 hand-code migration。

值得做的实验必须包含“byte size 一样但 meaning 变化”的 schema change。如果 BTF-aware checker 只会抓 ordinary property check 已经能发现的问题，它没有多少增量价值。

### per-link atomic replacement 不等于 multi-hook commit

`BPF_LINK_UPDATE` 对一个 link 已经是很强的 primitive。gap 只有在 application 真的存在 cross-hook invariant 时才出现，例如“ingress classifier 与 cgroup policy 必须使用同一个 policy epoch”。

缺的是 common generation gate，或者更底层的 multi-object commit primitive。sequential update 会产生一个很短但真实的 mixed-generation window。这个窗口是否重要必须靠 workload 判断。

正确的实验不是只量 update syscall latency，而是升级期间持续注入 traffic/event 和 failure，看是否有 event 被一个禁止的 generation combination 处理。如果现实 workload 根本观察不到有害 mixed state，那么额外 transaction machinery 就没有必要。

### persistent BPF object graph 的 crash recovery 语义还不够明确

pin 的目的就是让 object 跨过 controller restart，但这意味着 partially prepared generation 也可能跟着活下来。

缺的是 durable transaction journal 和 idempotent recovery rule。否则最需要 rollback 的时候，恰好也是 in-memory intent 消失的时候。

这个问题很好测：在 upgrade state machine 每个 side effect 之后 kill controller，再 restart，检查系统是否总能收敛到 old committed generation 或 new committed generation，而不会删除 live state。

## Promising directions with academic and production value

### 给 libbpf application 做 generation-gated upgrade runtime

**Gap.** 现有 primitive 能 prepare program，也能逐个替换 link，但 multi-hook application 没有一个逻辑 activation decision。

**Mechanism.** 在支持的 hook 上长期保留一个很小的 stable dispatcher。dispatcher 读取 application-wide generation selector，再跳到该 generation 对应的 program slot。new programs 和 maps 先 load 到 versioned namespace，旧 generation 继续运行。state 准备和 validation 完成后，controller 只更新 generation selector。old generation 一直保留到 retirement condition 满足。对于 map state，可以在 map type 允许的情况下使用与 generation 对齐的 versioned indirection。

这个设计必须明确 portability boundary。tail call 与 dispatcher 在不同 program type/hook 上能力不同，所以 runtime 应该公开：哪些 hook 能得到真正 generation gate，哪些只能 sequential `BPF_LINK_UPDATE`，哪些没有 kernel support 就无法达到 requested guarantee。

**Delta.** 和普通 libxdp dispatcher 或 program array 的区别不在“有 dispatcher”，而在 lifecycle protocol。program routing、state version、recovery metadata、retirement 都引用同一个 application generation。

**Artifact.** 一个 libbpf-based controller + manifest，先支持 XDP/TC，再支持一个 tracing 或 cgroup family；同时给 [bpftime](https://eunomia.dev/zh/bpftime/) 做 adapter，用同一个 generation protocol 测 userspace backend。

**Evaluation.** 用 stateless/stateful networking、policy、observability application，分别组合 1、2、4、8 个 coordinated hooks，在持续 packet/event load 下反复升级。每个 prepare、migration、selector、retirement step 前后都 fault inject。测 forbidden mixed-generation observation、lost event/packet、cutover latency、dispatcher steady-state overhead、rollback window 内 memory amplification 与 recovery time。baseline 包括 sequential link update、stop-and-restart 和 application-specific orchestration。

**Academic value.** 核心问题是：一个 generation selector 加明确 retirement，能不能在不同 BPF hook lifetime 上提供足够有用的 cross-object consistency，以及 abstraction 在哪里失效。

**Production value.** 平台团队可以复用同一个 update state machine，而不是每个 BPF controller 都重新实现 prepare/revert/finalize。manifest 还能直接驱动 health check、rollback 与 GC。

**Failure condition.** 如果真实 application 很少存在 cross-hook invariant，或者 dispatcher overhead 与 hook coverage 限制比短暂 sequential-update window 更贵，就不应该用这个 runtime 取代简单 link update。

### BTF-aware state migration，加显式 invariant

**Gap.** compatible pinned map 可以高效 reuse，但 structural compatibility 不足以覆盖 semantic schema evolution。

**Mechanism.** 每个 persistent state object 都带一个由 BTF + application metadata 定义的 schema version。load 前比较 old/new schema，把变化分类成 reusable、mechanically transformable、需要 application converter。需要 transform 时先填充 shadow map，再验证 key preservation、monotonic counter、policy epoch consistency 等 invariant，最后才让新 generation 绑定该 map。write-heavy workload 则明确选择 snapshot-plus-delta 或短暂 quiescence，并把 pause/copy cost 量出来。

**Delta.** 这不是 general serialization system，而是专门处理 BPF map schema 与 generation transition。它也超出普通 map property check：migration 成功和 invariant validation 会变成 activation 的前置条件。

**Artifact.** BTF schema differ、migration-plan generator、converter API 与 fault-injection harness，再从开源 BPF project 收集一组真实 schema change corpus，只保留 public-safe 的结构信息。

**Evaluation.** 覆盖 field add/rename、widen/narrow、等长 semantic change、map-type change、per-CPU state、LRU map，以及大到无法便宜 copy 的 map。比较 full copy、lazy migration、snapshot-plus-delta、explicit reset，测 migration throughput、pause time、memory amplification、update loss、invariant violation 和 developer effort。最重要的 correctness metric 不是“自动迁移越多越好”，而是 unsafe migration 能不能被拒绝。

**Academic value.** 它研究 long-lived eBPF state 哪些 semantic property 能从 type 推断、哪些必须由 application 明确声明，这个边界不是 verifier memory safety 能回答的。

**Production value.** operator 在 deploy 前就能知道这是 safe zero-downtime upgrade，还是必须接受 planned state loss / maintenance window。

**Failure condition.** 如果 corpus 显示绝大多数生产 map change 要么 exact reuse、要么本来就可以丢弃，那么做复杂 migration framework 没有必要。

### 给 pinned BPF object graph 做 crash-consistent journal

**Gap.** pin 会让 object 跨 controller restart 存活，但 controller memory 里的 upgrade intent 可能消失，此时 old/new generation 同时存在。

**Mechanism.** 保存一个很小的 durable transaction record：expected old generation、prepared new generation、object identity、migration status、commit decision、retirement status。每个 phase 必须 idempotent。recovery 按确定规则执行：commit 前可以丢弃或继续 prepared generation；commit 后恢复 new controller view 并完成 retirement；如果 active generation 无法确定，则 fail closed 并保留两组 object 供诊断，不允许猜。

journal 不一定放在 bpffs。如果 deployment 已经有 durable control-plane storage，用现有 storage 也可以。关键是 pinned object reference 和 transaction metadata 必须显式 reconcile。

**Delta.** Cilium 这类 production project 已经有 revert stack 和 finalizer。这里的贡献必须是一个最小 portable recovery state machine，加可以验证的 crash-consistency property，而不是另一套 project-specific callback。

**Artifact.** transaction library、recovery checker 与 deterministic fault injector，在每个 externally visible operation 之后 kill controller，并通过 `bpftool` compatible ID/pin 暴露 object graph 方便检查。

**Evaluation.** 随机跑数千次 upgrade，在每个 transition kill controller，包括 recovery 过程中再次 crash。验证每次都收敛到 valid committed generation，不会删 active map，并最终收集 unreachable objects。baseline 是没有 explicit journal 的简单 desired-state reconciler。

**Academic value.** 研究 reference-counted kernel objects + 小型 userspace journal 是否足够提供 crash-consistent application upgrade，还是必须引入更强 kernel transaction support。

**Production value.** 它针对最难排查的一类 operational failure：agent restart 以后 datapath 还在跑，但没人知道 persistent state 到底属于旧软件还是新软件。

**Failure condition.** 如果 idempotent desired-state reconciler 在同样 object graph 上能以更少 metadata 达到同样 correctness，那么 journal 是多余的，应该选更简单的 reconciler。

## 不应该一开始就假设必须新增 kernel multi-object transaction

很容易想到一个新的 syscall：传入多个 links、maps，一次原子 publish 全部 replacement。这在一些 stable dispatcher 不可行或开销太高的 hook 上，未来可能确实有价值。

但它不应该成为第一版方案。kernel 必须先回答不同 hook type 的 synchronization/lifetime 怎么组合、state migration 如何参加 transaction、expected revision 如何阻止 concurrent controller 提交 stale state，以及 failure 如何返回而不留下半安装 reference。一个 syscall 可以让 pointer swap 原子化，却不能自动让 application state transformation 正确。

更合理的 research path 是先用现有 primitive 在 userspace 实现 transaction model，收集哪些 case 无法满足 guarantee，再用这些 case 去 justify 最小 kernel primitive。最后真正需要的可能是 multi-link expected-generation commit、reusable generation handle，或者比“通用 BPF transaction syscall”窄得多的东西。

## What would change this conclusion?

最强的反方其实很简单：大部分 eBPF application 可能根本不需要 transactional upgrade。stateless tracer 只有一个 link，直接 `BPF_LINK_UPDATE`。map schema 稳定，就继续 reuse。能够接受短 maintenance window 的 application，可以 stop、migrate、restart，系统复杂度低很多。

三类结果会明显削弱本文结论。第一，production BPF deployment corpus 显示 coupled multi-hook upgrade 和 semantic map migration 非常罕见。第二，fault injection 说明 sequential link update + 普通 desired-state reconciliation 在现实 workload 中从不暴露有害 mixed generation。第三，generation dispatch 的 steady-state cost 或 hook-specific complexity 高到 operator 宁愿接受一个明确的小停顿。

相反，如果多个 project 都独立实现相似的 prepare/revert/finalize state machine，真实 incident 反复来自 program/state partial update，或者实验显示一个 generation protocol 可以低成本消除 policy/state inconsistency，那么 application-level transaction 的必要性就会变强。

因此本文不是在说“eBPF 需要数据库事务”。更准确的边界是：**当多个 persistent BPF object 共同定义一个 correctness invariant 时，upgrade mechanism 的 commit scope 也必须和这个 invariant 一样大**。Linux 已经给了我们大部分对象级 primitive。真正值得继续做的系统问题，是上面这一层究竟能有多小。

## References

- [Linux kernel: eBPF syscall reference](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [Linux kernel: libbpf overview and BPF application lifecycle](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html)
- [Linux kernel: map of maps](https://docs.kernel.org/bpf/map_of_maps.html)
- [libbpf-rs discussion: reusing an existing map during BPF code upgrades](https://github.com/libbpf/libbpf-rs/issues/52)
- [Cilium issue #38998: empty policy map after endpoint regeneration failure](https://github.com/cilium/cilium/issues/38998)
- [Cilium issue #19091: pinned map property mismatch and expected data loss](https://github.com/cilium/cilium/issues/19091)
- [bpftime: userspace eBPF runtime](https://github.com/eunomia-bpf/bpftime)
