---
date: 2026-09-21
slug: ebpf-link-controller-reconciliation
title: "eBPF Link 比控制器活得更久，安全吗？"
description: "Pinned BPF link 可以跨过控制器退出继续存在，但重启后的控制器可能丢失所有权信息并重复挂载。本文讨论可恢复的 link ownership、隔离式 reconciliation 和 crash-fuzz 验证。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - Operations
  - Kubernetes
research_question: "当 eBPF 控制器重启后，怎样用一套明确的 ownership 与 reconciliation contract，判断现存 link 应该接管、替换还是 detach，同时避免重复挂载或误删其他控制器的状态？"
source_cutoff: 2026-09-21
status: daily-report
---

# eBPF Link 比控制器活得更久，安全吗？

BPF link 有一个非常有用、但也很容易被误解的特性：它可以比创建它的 userspace 进程活得更久。

这正是 pinning 的用途。当前 libbpf API 对 [`bpf_link__pin()`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.h) 的说明很直接：pin 会增加 link 的 reference count，因此即使创建它的进程已经退出，link 仍然可以留在内核中。Linux 的 BPF UAPI 也是同一套 object lifetime 模型：只有当 fd、pin 以及其他引用都消失后，对象才会真正被释放。

对于一个短命令行工具，这通常只是便利功能。但对于长期运行的 controller，这会马上变成 recovery 问题。一次 crash、升级、容器重启或者不完整 cleanup 之后，新进程看到一个已经存在的 kernel link 时，需要回答：

- 这是我现在仍然想要的那个 attachment 吗？
- 它是不是旧 generation，应该被替换？
- 它是不是上一个 controller instance 留下来的 orphan？
- 它是不是另一个 controller 所有？
- 还是 userspace 的 metadata 已经消失，但 kernel object 本身仍然完全有效？

内核可以告诉 userspace 哪些 link 还活着，但它并不知道这些 link 在 controller 的 desired state 里分别代表什么。

Cilium 最近一个真实 regression 把这个边界表现得很清楚。在 [Cilium issue #46065](https://github.com/cilium/cilium/issues/46065) 中，agent 重启之后，`cgroup_inet_sock_release` BPF link 可能还留在内核里，但重启逻辑依赖的 pin file 已经不存在。连续重启会积累重复 link。报告中甚至出现了一个很危险的现象：agent readiness 和 `cilium status` 都显示正常，但 endpoint 长时间停在恢复状态，并出现 `policy_denied` 流量。这个 issue 在 2026 年 9 月 21 日被 stale bot 以 not planned 关闭；之前尝试清理 orphan link 的 PR [#46389](https://github.com/cilium/cilium/pull/46389) 也已经关闭，并没有 merge。

这并不是说 BPF link 的设计有问题。恰恰相反，kernel object 继续存在，是因为它仍然有合法 lifetime reference。真正出错的是 controller 对 kernel reality 的重建。

所以问题比一个 Cilium bug 更一般：**当 attachment lifetime 本来就故意和 process lifetime 解耦时，eBPF control plane 到底应该用什么 contract 来恢复 ownership？**

<!-- more -->

## Pin 是 lifetime reference，不是 ownership proof

最自然的实现方式，是把 bpffs path 当成 controller 的 source of truth：

```text
pin 存在     -> attachment 存在，而且是我的
pin 不存在   -> attachment 不存在
```

这个模型太强了。

BPF object 是 reference counted。pin 只是其中一个 reference，打开的 fd 是另一个，一些 attachment mechanism 或 object relationship 还可能保留额外引用。一个 userspace-visible name 被删掉，并不能证明对应 kernel object 已经消失。

反过来也一样。pin 还在，也不代表 controller 仍然希望这个 attachment 存在。内核并不知道某个 deployment generation 已经 rollback，不知道 Kubernetes object 已经删除，也不知道另一个 controller instance 已经接管了这个 hook。它只知道 object 还被引用。

因此 bpffs 是很好用的 durable handle，但并不是完整的 ownership ledger。

Linux 已经提供了足够多的 introspection primitive。当前 BPF syscall UAPI 有 [`BPF_LINK_GET_NEXT_ID`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)、`BPF_LINK_GET_FD_BY_ID` 和 `BPF_OBJ_GET_INFO_BY_FD`。libbpf 也有对应的 `bpf_link_get_next_id()`、`bpf_link_get_info_by_fd()` 等接口。在支持的 attach point 上，`BPF_PROG_QUERY` 还能返回 link IDs。

这些接口可以回答：

```text
现在有哪些 BPF link？
这个 link 指向哪个 program？
它是什么 link type？
这个 link 暴露了哪些 attach target 信息？
```

但它们不能直接回答：

```text
哪个 deployment generation 创建了它？
现在应该由哪个 controller 负责？
这是已经 commit 的 link，还是 controller setup 到一半就死了？
pin 不见了，到底代表对象删除，还是只代表 bookkeeping 丢了？
我 detach 它，会不会把另一个 owner 的 datapath 一起删掉？
```

restart bug 往往就出在这一层。

## Cilium 这个故障说明：existence 和 ownership 是两种状态

Cilium 报告里最值得注意的，并不是“旧 link 没有消失”。persistent attachment 本来就应该能够存活。

真正的问题更接近下面这个状态转换：

```text
旧 controller
    创建 / 保留 BPF link
    |
    | controller 重启，userspace bookkeeping 改变
    v
新 controller
    找不到预期 pin
    误认为没有自己拥有的 link
    创建新 link，或者走了不匹配的 detach path
    |
    v
kernel
    原来的 bpf_link 仍然存在
```

issue reporter 观察到，多次 restart 后旧 link 会不断积累。手工执行 `bpftool link detach id ...` 清理 stale attachment 后，下一轮 reconciliation 才能正常完成。

那份没有 merge 的 Cilium PR 也很有启发性。它的描述直接指出：pin file 已经被删除时，`bpf_link` 本身仍然可能留在 kernel 中。patch 尝试先 query cgroup 中通过 bpf_link attach 的 program，然后按 `LinkID` 清理 orphan link，再创建新 attachment。

作为局部修复，这很合理。但它也暴露出更一般的问题：如果逻辑退化成“没看到 pin，那就 detach 所有匹配 link”，只有在 controller 能证明所有匹配 link 都属于自己时才安全。共享 hook、多 controller、rolling upgrade，或者 attach mechanism 迁移时，这个证明并不简单。

所以 robust controller 至少需要三种状态，而不是两种：

```text
owned and desired
owned but undesired
present but ownership is uncertain
```

第三种状态不能默认折叠成“删掉”，也不能默认折叠成“忽略”。

## 这和 atomic eBPF upgrade 不是同一个问题

之前的 [stateful eBPF transactional upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/) 讨论的是：一个由多个 program、map、link 组成的应用，怎样从一个 committed generation 原子地切到下一个 generation，避免暴露 half-upgraded datapath。

restart reconciliation 的 failure model 不同。旧 controller 的 in-memory transaction state 可能已经完全丢失，新进程只能在事后醒来，重新推断 kernel 现在到底在运行什么。

同样，[kernel capability admission](https://eunomia.dev/research/ebpf-kernel-capability-evidence/) 关心 artifact 能不能在目标 kernel 上加载；[cross-kernel semantic compatibility](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/) 关心加载成功后，kernel upgrade 是否改变应用语义。

这里 kernel 可以完全没变，program 也可以完全合法。失败的是 control-plane identity：

```text
actual attachment set != controller 重建出来的 ownership model
```

这个边界现在越来越重要，因为 BPF link 已经逐渐成为 persistent、可更新 attachment state 的正常表达方式。link 越容易跨 process 存活，restart protocol 就越不能靠猜。

## 现有研究还缺什么

Linux 已经提供了很强的 object-lifetime primitive，但 controller-level lifecycle semantics 基本还是 application policy。

第一，link ID 是 inspection handle，不是 durable application identity。它可以唯一标识一个当前还活着的 kernel object，却不会编码 deployment、tenant、controller generation 或 desired-state key。

第二，bpffs path 是 durable name，却不能完整证明 kernel state。path 丢失时 link 可能仍然存活；path 存在时 desired state 也可能早就改变。

第三，只比较 program identity 往往不够。两个 controller 完全可能有意把同一份 program image attach 到同一个支持 multi-attach 的 target，但 ownership 不同。program tag 或 object digest 能证明“代码是否相同”，不能证明“责任是否相同”。

第四，health check 往往只观察 controller process，而不是 reconciliation 是否已经收敛。Cilium 这个案例特别有价值，就是因为 readiness 和 status 仍然正常，但 datapath orchestrator 一直失败，endpoint 还停在 restore 状态。

最后，很多 test suite 会测试 create、attach、update、detach 的正常路径，却不会在每一个 side effect 之后 kill controller，再检查下一次启动是否能精确恢复成一个 desired attachment，同时保证 foreign state 不被删除。

缺少的并不是另一套 attach API，而是一套建立在现有 kernel object 之上的 ownership-aware reconciliation contract。

## 研究方向一：给 attachment 一个跨重启的 generation-scoped receipt

第一个方向，是给每个 logical attachment 一个和当前 kernel link ID 分离的 durable identity。

**Gap。** controller 重启后可以 enumerate kernel link，但还需要把这些 link 映射回 logical desired-state record。pin path、program tag 和 link ID 都只能提供映射的一部分。

**机制。** 为 attachment 定义一个 stable logical key 和 generation：

```text
attachment_key: socketlb/cgroup-inet-sock-release
owner_scope: node-agent
controller_generation: 417
kernel_boot_id: ...
target_identity: cgroup + attach_type + target generation
program_identity: object digest + program name/tag
link_id: 1851
pin_path: /sys/fs/bpf/.../link
state: prepared | committed | retiring
```

第一版完全可以只在 userspace 做。controller 可以把 receipt 放在 bpffs object 旁边，或者写进一个小的 pinned metadata map。最重要的规则是：`link_id` 和 `pin_path` 是 logical key 的 evidence，不是 logical key 本身。

`kernel_boot_id` 可以避免 reboot 后旧 userspace receipt 误指向数值上被复用的新 kernel object。对于 cgroup、netdev 等可能被删除再创建的 target，也需要 target generation，不能只记一个看起来没变的 userspace name。

restart 时，controller 读取 receipt，再通过 `BPF_LINK_GET_NEXT_ID` 或 target-specific query enumerate live link，读取 `bpf_link_info`，然后尝试完成三方 join：

```text
desired attachment
    <-> durable receipt
    <-> live kernel link
```

任何缺失的边，都变成明确 recovery case，而不是隐含假设。

如果 userspace prototype 证明这个模型有价值，未来 kernel 可以考虑为 generic link info 提供一个 immutable opaque reconciliation tag。这个 tag 只提供 metadata，不产生新 authority，也不绕过 verifier。第一版并不需要改 UAPI。

**相对现状的变化。** 这比“记住 pin path”更强，也比通用 deployment manifest 更具体。它明确表达一个跨 process loss 仍然必须存在的 control-plane attachment identity。

**Prototype。** 做一个小型 libbpf controller，同时管理 cgroup 和 TCX link。正常使用 bpffs pin，同时故意删除部分 pin，但保留其他 kernel reference，验证 receipt 是否仍能找到并正确接管 link。

**Evaluation。** 统计 crash/restart 情况下的正确 adoption rate、false orphan classification 和误删 foreign link 次数。再加入两个 controller，有意把相同 program image attach 到同一个 multi-attach hook，测试 ownership 是否仍能区分。

**学术价值。** 把 BPF attachment recovery 变成可以定义 invariant、测量错误率的 identity-reconciliation 问题，而不是一组 ad hoc cleanup rule。

**生产价值。** operator 能明确回答“为什么这次 restart 接管了这个 link，或者为什么 detach 了它”，也能把同一个 stable key 用在 log、alert 和 rollback 工具里。

**失败条件。** 如果现有 pin path 加 link metadata 已经能在真实 crash、multi-controller hook 和 target recreation 中稳定唯一地恢复 ownership，那么 receipt layer 就没有必要。

## 研究方向二：ownership 不确定时先 quarantine，而不是直接 detach

第二个方向，是把 uncertainty 变成 reconciliation 的 first-class state。

**Gap。** restart routine 通常希望尽快收敛：删除旧 object，再重建 desired set。但 ownership evidence 不完整时，这很危险。留一个 duplicate link 可能是错的，误删 foreign 或仍然 authoritative 的 link 通常更糟。

**机制。** 在任何 mutation 之前，先把 observed link 分类：

```text
EXACT        receipt + target + program + generation 全部匹配
STALE        owned receipt 能证明这是旧 generation
MISSING      desired receipt 找不到 live link
FOREIGN      evidence 指向另一个 owner
AMBIGUOUS    link 存在，但 ownership proof 不完整
```

只有 `STALE` 可以立即进入 automatic detach。`EXACT` 直接接管，`MISSING` 重新创建，`FOREIGN` 保留，`AMBIGUOUS` 进入 quarantine path，继续收集 evidence 后再决定。

如果某个 hook 上 duplicate execution 本身不安全，那么 quarantine 还应该影响 readiness。controller 已经知道可能存在一个额外 attachment 时，就不应该把 datapath convergence 报成 healthy。

一个实际的 recovery sequence 可以是：

1. snapshot 当前 target 的 link set，并在有 query revision 时一起记录；
2. 不做 mutation，先完成分类；
3. 创建确实缺失的 candidate attachment；
4. 验证 desired program 已经在正确 target 上 active；
5. 只 detach ownership 已经证明为 stale 的 link；
6. 再 query 一次，只有 desired set 稳定后才宣布 ready。

这不是另一套 multi-object upgrade transaction，而是 old process 已经消失后，controller 用来重建 ownership 的 restart protocol。

**相对现状的变化。** 很多 cleanup logic 会把“pin 缺失”或者一次 lookup mismatch 当成足够证据，直接 recreate 或 delete。quarantine 要求 destructive cleanup 必须有 positive evidence，并把无法消除的 ambiguity 暴露给 health status。

**Prototype。** 先从 cgroup link 开始，因为 `BPF_PROG_QUERY` 可以暴露 attached program 和 link ID。模型稳定后，再扩展到 target metadata 不同的 TCX 和 tracing link。

**Evaluation。** 注入 missing pin、stale receipt、duplicate link、两个 controller overlap、cgroup path 被删除后重建，以及 concurrent attach/update。安全指标是误删 valid foreign link 的次数；liveness 指标是恢复到 exactly desired set 需要的时间。

**学术价值。** 这让 partial knowledge 下的 control-plane reconciliation 有明确的 safety/liveness tradeoff，可以系统实验，而不是只靠经验写 cleanup code。

**生产价值。** controller 在 ownership 有歧义时可以 fail closed，而不是在 duplicate attachment 和 destructive cleanup 之间来回震荡。

**失败条件。** 如果 ambiguity 太常见，导致 quarantine 经常阻断恢复；或者现有 kernel metadata 根本无法在合理时间内消除歧义，那么就需要更强的 kernel-visible ownership primitive。

## 研究方向三：crash-fuzz link lifecycle，而不只是测 datapath

第三个方向不是新 attach mechanism，而是一套专门测试 controller lifecycle protocol 的 evaluation system。

**Gap。** datapath packet test 可以全绿，但 controller recovery protocol 仍然是错的。问题只会在进程恰好死在两个 lifecycle operation 中间时出现。

**机制。** 做一个 deterministic crash-fuzz harness，在每个重要 side effect 后设置 failure point：

```text
create program
create link
pin link
write receipt
mark committed
unlink pin
update program
start retirement
detach old link
delete receipt
```

每个 failure point 都直接 kill controller，不执行 graceful cleanup。重启后比较三份状态：

- configuration 定义的 desired attachment state；
- kernel enumeration 与 target query 得到的 actual link state；
- hook 实际执行行为，包括到底跑了几个 program、执行顺序是什么。

最后一份不能省。两个 metadata 看起来很相似的 link，如果同时执行，仍然可能改变 policy、counter、socket state 或 packet mutation。

harness 还应该加入两个 controller instance，注入 overlapping restart。对于 cgroup multi-attach，需要区分合法 multi-program composition 和意外 duplicate ownership。

一个最小 convergence oracle 可以写成：

```text
对每个 logical attachment key：
    exactly one intended owner-generation is active

对每个 active owned link：
    都必须有 committed desired-state key 可以解释

对每个 foreign link：
    reconciliation 永远不能 detach
```

readiness 也应该进入 oracle。controller 如果在这些 invariant 成立之前就报告 healthy，即使最终能恢复，也算 test failure。

**相对现状的变化。** kernel BPF selftest 很擅长验证 object 和 attach semantics。这里测的是 userspace lifecycle protocol 在真实 kernel persistence 与 controller crash 下是否正确。

**Prototype。** 先做一个很小的 libbpf daemon，跑在 VM 里，只管理一个 cgroup target 和一个 counter program。duplicate execution 可以直接从 counter 看出来。oracle 稳定后再增加 TCX 和 tracing target。

**Evaluation。** 比较三种 controller：只看 pin path 的 recovery、enumerate-and-delete recovery，以及 receipt-plus-quarantine recovery。跑数千个 crash point，统计 false detach、duplicate execution 持续时间、recovery latency 和错误 readiness。

**学术价值。** 这会得到一个可复现的 persistent-kernel-object reconciliation benchmark。类似问题其实也存在于 networking、storage 和 device control plane，并不只属于 BPF。

**生产价值。** 同一套 harness 可以直接成为管理 persistent BPF state 的 agent 在发布和升级前的 regression gate。

**失败条件。** 如果现有 integration test 已经覆盖同等数量的 crash point，并且有同等严格的 kernel-state oracle，那么单独做这套 harness 就是重复建设。

## 今天部署时应该怎么做

对于当前 production controller，最保守的一条规则是：**不要把 missing pin 当成“link 不存在”的证明。**

restart 后，除了检查 bpffs，也要检查实际 attach point。根据 attach type 使用 `bpftool link show`、`BPF_LINK_GET_NEXT_ID`、`BPF_LINK_GET_FD_BY_ID`、`bpf_link_info` 和 target-specific query。保留足够的 durable metadata，能够说明某个 live link 为什么属于某个 desired attachment。

在 shared hook 上，如果 ownership 没有证明，不要因为 program 看起来匹配就自动全部 detach。program identity 和 ownership identity 是两件事。

把 reconciliation convergence 纳入 health。process 活着，不代表 datapath 已经恢复，更不代表没有 duplicate attachment。如果 controller 已经知道 attachment state 有歧义，就应该明确暴露，而不是继续返回 generic healthy。

最后，把 restart 当成 fault test，而不是只测试 graceful shutdown。分别在 link create、pin、unlink 和 replace 中间 kill controller。最有价值的 case，往往正是 userspace bookkeeping 与 kernel lifetime 不一致的时候。

如果需要先熟悉底层 primitive，可以配合阅读现有的 [detach 教程](https://eunomia.dev/tutorials/28-detach/)。但 production control plane 需要在这些 primitive 外面再加一层 ownership protocol。

## 什么证据会改变这个结论？

本文的核心结论是：persistent BPF link 除了 kernel-level lifetime semantics，还需要 controller-level ownership 和 reconciliation semantics。

有三类证据会削弱这个结论。

第一，如果未来 generic BPF link metadata 本身提供了稳定、足够 expressive 的 owner identity，并且真实 controller 能直接用它跨 crash、rolling restart 恢复，那么 userspace receipt 可以大幅简化甚至消失。

第二，如果生产数据证明 pin path 加现有 target query 已经可以在 shared hook、process crash 和 target recreation 中无歧义地恢复 ownership，那么问题会比 Cilium 这个故障表现出来的小很多。

第三，如果大规模 crash-injection 证明 duplicate 或 foreign-link failure mode 很少出现，而且所有支持 hook 在重复 attachment 下都天然 idempotent，那么 quarantine 的安全收益就不够高。

目前这三点都没有成立。Linux 已经提供了 durable BPF link object 和很好的 introspection primitive。真正还没有明确下来的，是创建它的 controller 已经消失后，怎样把这些 kernel object 重新解释成唯一、正确的 desired attachment set。