---
date: 2026-09-18
slug: ebpf-kernel-upgrade-semantic-compatibility
title: "内核升级后，同一个 eBPF 对象还能保持原来的语义吗？"
description: "CO-RE 能修正类型与字段偏移，但内核升级仍可能改变 verifier、kfunc、tracepoint 与 feature probe 的行为。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - Compatibility
  - BTF
  - CO-RE
research_question: "一个 eBPF 对象在内核升级前后都能成功 relocation、通过 verifier 并完成 attach 时，部署系统应该如何证明它仍然保持应用真正依赖的可观察行为？"
source_cutoff: 2026-09-18
status: daily-report
---

# 内核升级后，同一个 eBPF 对象还能保持原来的语义吗？

假设一个 eBPF object 在内核 A 上正常运行。机器升级到内核 B 后，同一份 object 仍然能完成 CO-RE relocation，通过 verifier，也能 attach 到目标位置。我们能不能据此说，这个 eBPF 应用已经兼容新内核？

不能直接这样下结论。

Linux 给 BPF 保留了一个有意稳定的核心 ABI，但真实应用经常依赖核心 ABI 以外的东西：tracepoint 的事件形状、可 attach 的内核函数、BTF 描述的内核内部类型、kfunc contract、verifier 规则，以及 loader 自己的 feature probe。CO-RE 可以把字段偏移修到新内核的布局，也可以查询某个类型或字段是否存在，但它无法自动证明“这个字段现在仍然代表应用认为的那件事”，更无法证明 loader 对一次 verifier 失败的解释在新内核上仍然成立。

2026 年 8 月的一个 Cilium 故障很能说明这个边界。Linux 7.2 改变了围绕 `bpf_set_retval` probe 的 verifier 行为后，Cilium 1.20 在启动阶段可能因为 feature detection 得到了不同的 verifier error，而把 probe 结果错误分类。这里变化的不只是版本号，甚至也不一定是 helper 本身是否存在，而是**用来推断 capability 的观察结果发生了变化**。

9 月 15 日的 [eBPF 内核 capability evidence 报告](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)讨论的是部署 admission：不要仅凭版本号猜一个 object 能不能在这台机器上运行。本文继续往下一层问：如果升级前后都能 load，怎么知道它做的还是同一件事？

<!-- more -->

## eBPF 内核升级兼容性不只是 CO-RE 能不能 relocation

CO-RE 解决的是一个明确而重要的问题。BPF object 中携带 BTF 和 CO-RE relocation record，loader 根据目标内核的 BTF 修改 BPF instruction 的 offset 或 immediate。当前 Linux 文档把 CO-RE relocation 大致分成 field、type 和 enum 三类，因此同一个 object 可以适配不少结构布局变化，而不需要为每个内核版本重新编译。

但这首先是**结构兼容性**，不是通用的语义等价证明。

假设一个程序通过 CO-RE 成功读到了某个内核字段，至少有三个不同的问题：

1. **字段存在，而且 instruction 能正确定位它。** CO-RE 往往可以回答。
2. **relocation 后的程序能通过 verifier。** 目标内核在 load 时回答。
3. **这个值现在仍然代表应用以为的那个系统事实。** 前两个步骤通常不能单独回答。

第三个问题才是 observability 和 policy 最容易出错的地方。一个 tracing 程序可能仍然能读取某个合法的整数，但内核更新这个字段的生命周期已经变化；一个 hook 仍然存在，却不再覆盖应用原本认为的完整操作；一个 kfunc 名字还在，但允许的调用上下文、pointer ownership 或参数约束已经发生变化。

因此真正应该区分的是：**structural adaptation 和 behavioral compatibility。**

它也不同于此前的 [eBPF 架构特化报告](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)。架构特化关心不同 JIT 或 native backend 是否保留同一份 BPF 语义；这里关心的是同一个 object 面对新的 kernel environment 时，那些赋予它实际含义的内核侧假设还是否成立。

## Linux 明确把一部分 BPF-facing interface 留在不稳定边界

Linux 的 BPF Design Q&A 对这个问题划得很清楚。BPF instruction、program argument、helper 及其参数、已定义的 return code 属于稳定 ABI。但生产 BPF 应用大量使用的另外一些接口并不属于这个承诺。

Tracepoint 不是 stable ABI。kprobe 可以 attach 的内核函数位置也不是 stable ABI。tracing 程序读取的 kernel internal data structure 本来就会随着内核演化。Linux 文档建议使用 CO-RE 降低这些变化带来的适配成本，但这并不会把内部 kernel function 变成永久不变的公共接口。

kfunc 更直接。当前内核文档明确说明：kfunc 不像普通 BPF helper 那样具有稳定接口，它可以在不同 kernel release 之间变化。某个 kfunc 对哪些 program type 可见、verifier 如何判断 pointer 是否有效，这些规则也可能继续演进。

因此，一个真实的 compatibility statement 最好按依赖类型拆开：

```text
stable BPF ABI dependency
    -> 除非出现 kernel regression，否则预期保持兼容

CO-RE / BTF structural dependency
    -> relocation，并验证目标类型和字段证据

unstable tracepoint / kprobe / kfunc dependency
    -> 需要针对目标内核的新证据

loader / feature-probe dependency
    -> 验证 probe 的结果和解释方式仍然有效
```

只记录一个“load 成功”的 bit，会把这四层完全压扁。

## Feature probe 自己也会发生 drift

主动 probe 通常比 `uname -r` 好，因为它在问目标内核“你实际上支持什么”。但 probe 本身也是程序，而“如何解释 probe 结果”也可能形成新的兼容性边界。

Linux 7.2 上的 Cilium 故障就是一个很好的例子。Cilium 在检测 `CGroupSock` program type 下的 `FnSetRetval` 时失败，probe load 得到 `R1 is not a scalar` 的 verifier 信息。原有检测逻辑依赖另一种 failure shape 去区分“helper 存在”和“helper 不存在”，于是 verifier 行为变化打破了这个推断。

这不意味着 active probing 是错的。更准确的 contract 是：

> 一个 probe 只有在 loader 对它的解释方式同样适用于目标内核时，才算有效证据。

因此，probe 应尽量依赖 machine-readable outcome，而不是绑定 verifier log 的文本细节。如果必须通过特定 verifier rejection 来推断 helper、program type 或某个能力存在，那么 CI 应该把**这一份 probe implementation**本身放进支持内核矩阵里测试。

BTF tooling 也有同样的问题。2026 年 8 月，cilium/ebpf 修复过一个 essential-name indexing bug：像 `___pskb_trim` 这样以三个下划线开头的 kernel BTF function，因为 userspace 的名称归一化逻辑而无法通过 `TypeByName` 正常找到，从而影响 fentry/fexit attach。内核函数和 BTF 可以都存在，但 loader library 的规则仍然可能让它表现成“不支持”。

所以 compatibility 不是单独的 kernel 属性，而是 **kernel + BTF + loader/tooling + object** 整条路径的属性。

## 真正没被覆盖的 gap：admission 成功以后，行为是否仍然兼容

上一份 capability-evidence 报告提出了 artifact-bound receipt，用来解释某个具体 object 为什么在某台机器上被接受或拒绝。这一步很重要，但升级问题还没有结束。

假设升级前后都拿到了成功 receipt：

```text
kernel A: relocate pass -> verifier pass -> attach pass
kernel B: relocate pass -> verifier pass -> attach pass
```

这只能证明两个环境都接受了 artifact。它没有自动证明两个环境观察到了同样的事件、执行了同样的 policy boundary，或者在同样 workload 下产生了等价的状态变化。

今天已经有很多基础机制覆盖其中一部分：CO-RE 负责结构 relocation；verifier 做目标内核上的安全和 admissibility 检查；BPF selftests 覆盖大量内核行为与 regression；cilium/ebpf 一类项目会在多个 kernel version 上跑 CI 和 feature probe；生产环境也可以在大规模 rollout 前使用 canary。

缺少的是一个更直接的 deployment abstraction：**把一个 BPF artifact 与它必须保持的应用行为 contract 绑定，并且让这个 contract 明确跨过旧内核和新内核。**

这个 contract 不应该试图冻结所有 kernel internal。它只需要描述“这个 BPF 应用为什么有用”所依赖的 observable invariant。对于 network policy，可以是 packet verdict 和 map state transition；对于 profiler，可以是 event coverage 与 attribution relation；对于 tracing，可以是一个操作生命周期由哪些事件表示，以及若干字段之间必须满足的关系。

## 研究方向一：给 BPF artifact 配一组跨内核 semantic witness

第一个方向是把兼容测试从“内核总体通过”推进到“这个具体 artifact 的语义通过”。

**Gap。** 同一个 object 在两个内核上都能成功 load，但应用的 observation 或 policy assumption 仍可能不同。通用 BPF selftest 不知道一个具体应用真正需要保持什么。

**机制。** 在 BPF object 旁边生成一小组 semantic witness。每个 witness 包含可控制的 stimulus、必须成立的 observable invariant，以及明确的 tolerance。部署 harness 使用同一份 object 和同一代 loader，在旧内核与候选新内核上跑完全相同的 witness。

Witness 应该比较语义，而不是原始 timestamp 或无关实现细节。例如：

- 一条 network flow 必须得到相同的 allow/drop decision 和 policy-generation transition；
- 一个 file-operation sequence 必须产生一个逻辑上的 open lifecycle，即使内部函数已经变化；
- 一组 scheduler event 必须保持某个 task-state invariant；
- 一个 profiler workload 的 sample 数允许在范围内波动，但 parent/child causal attribution 必须一致。

输出也不应该只是一个 checksum，而应该像这样：

```text
artifact: sha256:...
loader: ...
old_kernel: ...
new_kernel: ...
witnesses:
  policy_allow: equivalent
  policy_revoke: equivalent
  event_lifecycle: changed
  attribution: equivalent-with-tolerance
```

**与现有做法的差异。** Kernel selftest 测 kernel；普通 unit test 测应用实现。这里的 witness 是一个 deployment artifact，显式跨两个 target kernel，并编码应用真正要求保存的语义。

**Prototype。** 先做 20 到 40 个小型 CO-RE 程序，覆盖 fentry/fexit、tracepoint、helper、kfunc、map，以及一条真实 network policy 路径。每个程序都带可重复的 namespace/VM stimulus 和 normalize 后的结果。

**Evaluation。** 使用 upstream LTS、current kernel、几种 distribution kernel，再加入故意改变 attach target、BTF layout、verifier rule 或 lifecycle timing 的 mutation。核心指标是：load-only gate 认为兼容但 witness 能抓到行为变化的比例，以及 witness 因过度绑定实现细节产生的 false alarm。

**学术价值。** 把“portable BPF”从 loadability claim 变成可以被证伪的 behavioral compatibility claim。

**生产价值。** Fleet kernel rollout 可以被一个具体 invariant 阻止，而不是只得到“unsupported kernel”这种模糊标签。

**失败条件。** 如果每个有用的 witness 都必须重现几乎整个应用，或者正常的无害 kernel refactor 会让 witness 大量失败，那么这个 abstraction 太贵，应该进一步收窄。

## 研究方向二：把 semantic drift 定位到真正变化的依赖

发现 witness 失败之后，下一步是回答“为什么”。

**Gap。** 升级后的差异可能来自 BTF、CO-RE、verifier、attach target、kfunc、loader probe、map behavior 或应用逻辑。现实中的 startup/runtime failure 往往把这些层堆在一起。

**机制。** 从 artifact 与 loader 构建 compatibility dependency graph。节点记录具体依赖，例如 CO-RE field relocation、program type、helper/kfunc signature、attach target、map feature、verifier-sensitive construct 和 loader probe；每个 witness 则连接到它实际依赖的节点。

升级时同时抓取两侧证据：

```text
witness changed
    |
    +-- target BTF digest changed
    +-- CO-RE resolution changed
    +-- verifier decision/log class changed
    +-- attach target identity changed
    +-- kfunc signature/visibility changed
    +-- feature probe outcome changed
```

系统随后报告“与该 witness failure 一致的最小 changed dependency set”，而不是只丢给工程师两份巨大的 verifier log。

**与 9 月 15 日 capability receipt 的差异。** Receipt 解释一次 admission decision；这个 graph 连接的是**两个都成功 admission 的环境之间出现的行为 regression**，并把它定位回发生变化的兼容证据。

**Prototype。** 扩展一个 libbpf-based loader，输出 normalize 后的 dependency record，并关联 witness ID。至少加入 target BTF hash、CO-RE relocation result、helper/kfunc availability、attach metadata 和 verifier outcome class。

**Evaluation。** 一次注入一个 controlled incompatibility，再注入两到三个组合变化。测 root-cause localization precision、诊断时间和仍然无法归因的 failure 比例，并与 kernel-version diff、raw verifier log、`bpftool feature` 全量 diff 比较。

**学术价值。** 验证 compatibility 是否可以建模为一个 causal dependency problem，而不是静态支持矩阵。

**生产价值。** Operator 能更快区分“需要重新编译 object”“更新 loader probe”“切换 tracepoint fallback”和“内核确实改变了事件生命周期”。

**失败条件。** 如果 dependency graph 最后变成另一套需要人工长期维护的 kernel model，或者大多数语义变化根本无法映射到可观察依赖，它就没有比传统 support table 好多少。

## 研究方向三：让 kernel upgrade 经过 eBPF semantic promotion gate

第三个方向是把上面的证据放进真实 rollout 流程。

**Gap。** Fleet kernel qualification 往往验证 boot、通用 workload 和 package compatibility；BPF 应用则单独验证“程序能 load”。两边都通过，并不等于 attach 后的 probe 和 policy 仍保持应用级语义。

**机制。** 在大规模部署前，把 candidate kernel 放到一小组代表性 VM 或 node 上。用当前已接受内核和候选内核分别跑 artifact witness 与 dependency capture。所有 hard invariant 一致，而且允许变化的指标仍在预先声明的 envelope 内，才允许继续 rollout。

Gate 不应该只给 pass/fail，而应该区分：

- **equivalent**：hard invariant 一致；
- **compatible with declared variance**：差异只在容许范围；
- **fallback required**：替代 attach/implementation 可以保持 contract；
- **behavioral regression**：object 能 load，但 hard witness 改变；
- **admission regression**：relocation、verifier 或 attach 失败；
- **probe regression**：application load 之前 feature inference 已经发生变化。

**Prototype。** 把 gate 接入 kernel-matrix CI 和 canary-node upgrader。旧/新 receipt、witness output 与 dependency delta 一起保存为 content-addressed promotion artifact。

**Evaluation。** Replay 历史 compatibility failure 和 synthetic mutation，包括 Linux 7.2 这类 feature-probe failure。比较四种策略：kernel-version allowlist、load-only gate、capability-receipt gate、semantic promotion gate。重点测 unsafe promotion、unnecessary block、diagnosis latency 和 qualification cost。

**学术价值。** 把 kernel/BPF compatibility 转换成可测的 false-admission 与 false-rejection 问题，而不是只维护一张静态支持表。

**生产价值。** 在升级到整个 fleet 之前，给 BPF regression 一个明确的拦截位置。

**失败条件。** 如果 canary kernel 和 witness workload 对 production behavior 的预测能力不足，无法减少 unsafe promotion，那么这个 gate 只是增加流程成本。

## 生产部署不需要每次 load 都把全部测试重跑一遍

Semantic gate 并不意味着每次 process start 都要起一套 compatibility lab。

重测试应该发生在 artifact build、kernel qualification 和受控 canary 阶段。Runtime load 可以继续使用前一份 [capability-evidence 报告](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/)里的缓存证据，按 kernel/BTF/loader/artifact identity 复用已经验证过的结果。

一个更实际的生产路径可以是：

```text
known-bad / unsupported kernel?
        -> reject

artifact capability receipt still valid?
        -> no: probe / relocate / load / attach

这个 kernel generation 已通过 artifact witnesses?
        -> yes: reuse qualified result
        -> no: broad rollout 前先做 upgrade qualification

发现 behavior difference?
        -> 使用已验证 fallback、暂停 rollout、或更新 artifact/loader
```

关键是不要让证据强度和 claim 强度错位。只有“CO-RE relocation 成功”，就只声称结构适配成功；真实 object load 通过，才能说 admission 通过；如果同一套 witness 确实跨旧/新内核验证了应用 invariant，才有理由做更强的 behavioral compatibility claim。

对于 kfunc、kernel tracing target 这类本来就会继续演进的接口，这种方法尤其合适。系统不用假装它们永远稳定，而是在变化时要求重新产生证据。

## 什么证据会改变本文结论？

有三类结果会削弱建立显式 semantic upgrade contract 的必要性。

第一，如果在足够广的 upstream、distribution kernel 和真实 BPF workload 上，研究发现“CO-RE 成功 + verifier/attach 成功”几乎总能预测应用级行为，false admission 可以忽略，那么 load-time evidence 可能已经足够强。

第二，如果真实世界的大多数 upgrade failure 都是清晰的 admission failure，而不是 silent behavior mismatch 或 probe-level mismatch，那么更完整的 capability receipt 也许就能覆盖大部分生产价值，不必再加跨内核 witness。

第三，如果 application-specific witness 太脆弱，无法区分无害 kernel implementation change 和真正影响语义的变化，那么共享 semantic gate 可能制造的 false block 比它避免的问题更多。

目前的一手证据更支持相反方向：Linux 有意同时保留稳定和不稳定的 BPF-facing interface；CO-RE 文档明确描述的是 relocation；而近期生产故障已经说明，甚至 feature probe 对 verifier 结果的解释也会随内核行为发生 drift。因此更安全的部署模型应该分层：**先适配结构，再证明 admission，最后跨升级边界验证应用真正依赖的语义。**

## 参考资料

- [Linux BPF Design Q&A](https://docs.kernel.org/bpf/bpf_design_QA.html)
- [Linux BPF LLVM Relocations 与 CO-RE relocation](https://docs.kernel.org/bpf/llvm_reloc.html)
- [Linux BPF Type Format 文档](https://docs.kernel.org/bpf/btf.html)
- [Linux BPF Kernel Functions (kfuncs)](https://docs.kernel.org/bpf/kfuncs.html)
- [cilium/ebpf issue #2084：Linux 7.2-rc6 feature probe failure](https://github.com/cilium/ebpf/issues/2084)
- [Cilium issue #48016：Linux 7.2 `bpf_set_retval` probe failure](https://github.com/cilium/cilium/issues/48016)
- [cilium/ebpf PR #2086：修复 triple-underscore BTF essential-name indexing](https://github.com/cilium/ebpf/pull/2086)
- [cilium/ebpf repository 与 kernel-version CI](https://github.com/cilium/ebpf)
