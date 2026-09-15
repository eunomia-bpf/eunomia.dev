---
date: 2026-09-15
slug: ebpf-kernel-capability-evidence
title: "eBPF 加载器能相信内核版本号吗？"
description: "内核版本号不能证明真实 eBPF 能力。本文提出与程序绑定的能力证据、语义 canary 和可复现支持范围。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - Compatibility
  - libbpf
  - BTF
research_question: "当发行版回移、内核配置、BTF、权限和快速演化的 BPF 接口让版本判断变得不完整时，eBPF 加载器应该怎样判断一台真实部署的内核究竟能安全加载和挂载什么？"
source_cutoff: 2026-09-15
status: daily-report
---

# eBPF 加载器能相信内核版本号吗？

很多 loader 做兼容性判断时，第一步会问一个很自然的问题：`uname -r` 是多少？

这个信息适合做机器清单，却不够当作 capability contract。

发行版可以长期保留一个较旧的 upstream base version，同时把后续多年的 subsystem 改动回移进去。一台机器还可能编译掉某个 BPF 功能、禁用 unprivileged BPF、暴露不同的 BTF surface，或者携带与同版本 upstream 不同的 verifier 和 kfunc 行为。即使两台机器看起来使用了相近的 kernel version，同一个 BPF object 也可能得到不同结果，因为真正决定兼容性的事实没有被压缩进一个版本整数里。

因此更准确的结论不是“永远不要检查版本”。版本和发行版 metadata 仍然适合定义支持范围、屏蔽已知坏版本和帮助排障。但如果 loader 真正要决定一份具体 BPF artifact 能不能运行，就应该优先相信**它即将使用的那台内核直接给出的证据**，而不是只根据这台内核看起来像哪个 upstream 版本来猜。

本文把这个原则发展成一个更明确的部署契约：loader 为每次关键兼容性决策生成一份可复现的 **capability receipt**。它把主动 feature probe、运行内核的 BTF、artifact 自己的 relocation/verifier 结果、实际权限上下文，以及在“存在”还不能证明“行为正确”时执行的小型 semantic canary 绑定在一起。它不仅解释为什么走 fast path，也解释为什么选择 fallback，或者为什么拒绝加载。

<!-- more -->

这和之前的[架构特定 eBPF 优化](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)问题不同。那篇文章研究的是一个可选 native optimization 在不同 CPU/JIT backend 上是否有资格启用，同时保留同一份 portable semantic witness。这里的 BPF object 本身不变，问题是现实 Linux 部署，尤其带发行版 backport 和本地安全策略的机器，究竟有没有提供 object 所依赖的 BPF interface。

它也不同于[有状态 eBPF 的事务式升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)。事务升级解决的是已经运行的多对象应用怎样在版本切换时不破坏状态。capability evidence 发生得更早：candidate object 到底能不能被这台 host 接受，支持这个结论的证据是什么？

## 发行版内核版本更像 lineage label，不是 feature bitmap

Red Hat 当前的 RHEL kernel 文档把这个问题说得很直接：RHEL kernel version 里的 upstream base number 并不会列出所有 source change，也不能只靠 version string 判断某个 upstream feature、API 或 driver behavior 是否存在。RHEL 会保留稳定 base，同时持续回移 bug fix、feature、hardware enablement 和发行版自己的改动。

RHEL 9.6 给出了很具体的例子。它的 kernel package 仍然以 `5.14.0` 为 base，但 release notes 明确写着 **eBPF facility 已经 rebase 到 Linux 6.12**。同一个版本还包含 BPF token、BPF arena、新 kfunc，以及检测 running kernel kfunc 的能力。如果 policy 写成 `kernel >= 6.12 才有 feature X`，它就会错误描述一台 uname 仍然以 5.14 开头、但 BPF subsystem 已经吸收大量更新实现的机器。

Red Hat 还专门发布完整的 “Available BPF features” 章节，而且内容由 `bpftool feature` 自动生成。这个做法本身就很有启发：发行版没有要求用户只靠 upstream base number 倒推 BPF 能力，而是把实际观察到的 kernel capability 列出来，包括 config、program type、helper、map type 等信息。

backport 也不只发生在 feature 上。安全和 correctness fix 同样会跨 version line 移动。比如 Ubuntu 对 CVE-2021-3490 的记录说明，一个 eBPF verifier 修复被回移到多个 stable kernel series。这里真正重要的是具体 patch history，而不是简单的 major/minor threshold。

这并不等于 version check 没价值。它仍然适合回答这些问题：

- vendor 是否正式支持这台 host；
- 某个已知 regression range 是否应该直接 block；
- package、kernel ABI 或 distribution release 属于哪个测试/支持 tier；
- 接下来应该收集哪类 compatibility evidence。

真正危险的是把 version string 当成直接证明，认为它已经回答了 `这个 program type + 这些 helper + 这个 kfunc + 这个 attach path + 当前 verifier behavior 是否存在`。

## Linux 和 libbpf 已经在很多地方选择主动探测，而不是版本表

`bpftool feature probe kernel` 会直接询问 running kernel。当前 bpftool 文档列出的内容包括 `bpf()` syscall、JIT 状态、program type、helper function 以及其他 BPF-related parameter。较新的 bpftool 还会把“工具编译时知道哪些 builtin”与“目标系统实际支持哪些能力”分开。

libbpf 也提供相同思路的 API。`libbpf_probe_bpf_prog_type()` 会尝试加载一个最小程序，判断 host kernel 是否支持某种 program type。`libbpf_probe_bpf_map_type()` 探测 map support。`libbpf_probe_bpf_helper()` 判断某个 helper 在指定 program type 下是否可用。这些 API 不是查一张用 `LINUX_VERSION_CODE` 索引的静态表，而是真的去问 kernel。

BTF 和 CO-RE 又提供了另一类证据。当内核配置支持时，running kernel 会在 `/sys/kernel/btf/vmlinux` 暴露权威 BTF。libbpf 可以拿 BPF object 里记录的 type/relocation 信息与 target BTF 做匹配，修正 field offset 和相关 type-dependent reference。于是 loader 可以根据目标机器真实的 type surface 工作，而不是假设某个版本的 `struct task_struct` 一定长成另一个版本的样子。

这些机制实际上已经形成一条 evidence ladder：

```text
version / distro metadata
        |
        v
broad active capability probes
        |
        v
target BTF + CO-RE relocation
        |
        v
object-specific verifier/load result
        |
        v
attach / semantic canary when needed
```

越往下，证据越直接回答 loader 真正关心的问题。

## “feature 存在”仍然不等于“这份 object 兼容”

general capability probe 可以告诉你 kernel 支持某个 program type 或 helper，但实际 object 仍然可能失败，因为还有不少边界只能结合具体程序判断。

### CO-RE 解决结构 relocation，不解决全部 BPF contract

CO-RE 非常擅长一个明确问题：根据 target BTF 让 kernel type/field reference 可 relocation。它并不会证明程序使用的每个 helper、map、kfunc、attach type、verifier rule 或 runtime semantic assumption 都仍然成立。

Linux 的 BPF design documentation 明确区分稳定 BPF ABI 与不稳定的 kernel internal/tracepoint。kfunc 也不同于稳定 helper。当前 kernel 文档说明，kfunc 属于 kernel-to-kernel API，没有 hard stability guarantee；在合理理由下 maintainer 可以修改或移除它，而且 kfunc visibility 还可以依 program type 不同。

随着越来越多的新 BPF 功能通过 kfunc 提供，这个边界只会更重要。loader 不能把“文档里存在这个 symbol”简化成“我的 object 在这里一定能调用它”。

### 权限本身就是 capability observation 的一部分

feature probing 还受权限影响。libbpf probe API 会提醒调用者，feature check 需要合适的 `CAP_*` 或 root。`bpftool feature` 专门有 `unprivileged` 模式，就是为了避免非 root 探测时把“我没权限看到”误判成“kernel 不支持”。

所以生产环境的 receipt 必须记录 **probe authority context**。`unsupported` 和 `这些 credentials 下无法观测` 是两种不同状态。同样，一个 privileged deployment controller 探测成功，也不应该拿自己的结果去承诺 unprivileged tenant 能执行同样操作。

### verifier 是一个可执行的 compatibility boundary

verifier 同时处理 program type、helper/kfunc contract、pointer/lifetime rule、BTF、kernel configuration 和具体 instruction graph。真正让 final object 成功 load，比 broad feature flag 更接近最终兼容性证明。

而且 BPF subsystem 在这里仍然持续变化。2026 年 9 月的一组 bpf-next patch 正在统一 helper 与 kfunc argument checking，其中涉及 type admission、nullability、memory、BTF、packet access 和 resource ownership。无论这组 patch 最终以什么形式合入，它至少说明 verifier-side contract 仍然是活跃实现面，而不是冻结的 lookup table。

因此对 loader 来说，verifier 更适合被当成需要保存结果的 admission oracle，而不是藏在 compatibility table 后面的麻烦步骤。

## 缺少的是一份和 artifact 绑定的 capability receipt

现有系统已经拥有很多零散证据，但常常没有把它们绑定成一个长期可解释的答案。

一份有用的 capability receipt 至少应该独立记录四类东西：

1. **Target identity。** Distribution/package identity、完整 kernel release、architecture、能获得时的 boot/build identity、相关 config evidence、BTF digest，以及 security/privilege context。
2. **Probe identity。** bpftool/libbpf version、probe mode、effective capability，以及 policy 实际使用的 general feature result。
3. **Artifact identity。** BPF object/skeleton hash，必要时记录 compiler/libbpf compatibility metadata，expected program/map/attach type、kfunc/helper dependency，以及 CO-RE relocation result。
4. **Admission result。** verifier/load outcome、规范化 error class、verifier-log digest 或按策略保存的完整 log、尝试过的 attach result、selected fallback，以及 semantic canary result。

例如：

```text
target:
  kernel_release: 5.14.0-...el9
  distro_package: ...
  btf_sha256: ...
  privilege_profile: deployment-controller-v2

artifact:
  object_sha256: ...
  requires:
    prog_types: [tracing]
    maps: [ringbuf]
    kfuncs: [...]

probe:
  bpftool: ...
  libbpf: ...
  general_features: ...
  core_relocations: pass

admission:
  verifier: pass
  attach_canary: pass
  selected_path: fentry
  fallback: tracepoint
```

这不是要把所有 kernel internal 冻结成一个新 ABI。它只是 evidence artifact。它要让部署系统在事故后能够回答：**当时我们为什么认为“这份 BPF artifact 在这台 host、这个权限上下文下是 compatible 的”？实际观察到了什么？**

## 当前工作还弱在哪里

### General feature matrix 比单个应用真正依赖的范围大得多

`bpftool feature` 可以生成很丰富的 host capability view，vendor release notes 也可以为某个 shipped kernel 发布完整矩阵。但一个应用通常只依赖很小的 conjunction：一个 program type、几个 helper/kfunc、少量 map type、某种 attach mechanism、若干 BTF type，再加上具体 verifier behavior。

大矩阵说明“host 总体能做什么”，却不能解释“这个 deployment decision 到底依赖哪些 row”，也不能说明缺一个 row 时哪个 fallback 才合法。

缺少的是 object-bound dependency projection：从 artifact 自动提取最小 relevant capability claim，测试它，并保存对应证据。

### Load success 仍然可能弱于 runtime-semantic compatibility

有些属性只有 attach 后，甚至只有相关 hook 真正运行时才会被触发。load-only test 可以证明 verifier acceptance，却不能证明目标 attachment 一定存在，也未必证明某个 BTF-backed kfunc 在这个精确 context 中可用，或者某个 distribution-specific behavior 符合应用假设。

缺少的是一种安全测试 semantic edge 的方法，又不能让 compatibility probing 本身产生危险 production side effect。

### Host 一旦升级，compatibility incident 很难复现

fleet rolling upgrade 后，一条 `object failed on kernel 5.14` 的记录通常太弱。到底是哪个 distribution build、config、BTF、权限、libbpf、verifier backport、kfunc contract 还是 object 本身？

如果没有 durable receipt，host reboot 或升级后关键证据就消失了。support matrix 可以告诉你“理论上应该工作”，却不能 replay loader 当时真实观察到了什么。

## 值得继续做的研究方向

### 1. 自动生成最小、与 artifact 绑定的 capability receipt

第一步可以在 loader 里加入 dependency extraction pass，在真正 load 前先提取 artifact 的 compatibility dependency。prototype 把 ELF/BTF/CO-RE metadata 与 loader 已知的 program、map、helper、kfunc、attach requirement 组合起来，只运行决定这份 artifact 所需要的 probes。

输出是一份 signed 或 content-addressed receipt，可以按 target identity cache，也可以挂到 deployment telemetry 上。

学术问题是：自动导出的较小 capability set，是否比 version table 或 broad distribution support matrix 更准确预测真实 load/attach success。生产价值是不用每次启动都 probe 全部 BPF feature，也能得到可解释 admission 和 deterministic fallback。

评测可以覆盖 upstream kernel 和长期维护、backport 很重的 distribution kernel，再加入多个 privilege profile。对每个 artifact/host pair 比较四种 predictor：version threshold、vendor support table、broad `bpftool feature` output、artifact-bound receipt。主要指标应该是 false admission 与 false rejection，而不是“发现了多少 feature”。

如果 derived dependency set 很不稳定，最终几乎和 full matrix 一样大，或者并没有显著降低 compatibility misclassification，这个方向就失败了。

### 2. 为模糊边界加入 side-effect-bounded semantic canary

当 general probe 加 object load 仍不足够时，可以运行一个只覆盖不确定边界的极小 canary。

map canary 可以 create/drop 目标 map type。kfunc canary 可以 load 一段在目标 program context 中调用该函数的最小合法程序。attach canary 只有在目标 hook 已证明受 namespace、temporary cgroup 或 test interface 的隔离边界约束时，才可以依赖这些环境；disposable link 只能限制生命周期，不能限制观测范围。像 fentry、tracepoint 这类可能观察 host-wide 事件的 hook 应该放到 dedicated test VM/host 中验证，或者不要在生产环境主动探测。每个 canary 都需要显式 side-effect budget 和 cleanup contract。

研究问题是怎样选择足够强、能预测真实应用，又足够安全便宜的 canary。它可以被建模成 **compatibility test selection**：已知 dependency graph 与历史 failure，挑选最小 probe set，把 deployment uncertainty 降到阈值以下。

评测矩阵应该主动制造 missing config、permission change、BTF difference、backported verifier behavior、absent attach target 和 changing kfunc surface，对比 load-only admission 与 load-plus-canary admission 的预测能力。

如果 canary 最终变成一套脆弱的“平行应用实现”、需要危险 production mutation，或者仍然漏掉运维真正关心的 failure class，这个方向就不成立。

### 3. 构建可 replay 的 artifact-to-kernel support envelope

单份 receipt 解释一次 decision。support envelope 则把 fleet 或 CI matrix 上的很多 receipt 聚合起来。

每个 BPF artifact generation 可以维护一组已测试 target identity 和 outcome，但 key 应该主要来自 observed capability evidence，而不是只有 `kernel >= X`。这样 envelope 可以回答：

- 哪些 target capability 真正 admission 过这份 object；
- 哪些 verifier/attach failure class 会重复出现；
- 哪些 fallback path 真的被执行过，而不是只在 source 里存在；
- distribution update 是否在 upstream base version 不变时改变了 capability evidence；
- 新 artifact generation 是扩大还是缩小了 proven support set。

研究 artifact 可以把 CI、production receipt 和 replay harness 连起来。把保存的 receipt 输入 synthetic target model 或 matching kernel VM，检查 loader 是否作出同样 decision。Linux BPF selftests 可以提供方法论上的先例：kernel BPF development 本身就更偏好由 bot 持续运行 selftest，用来覆盖 functional/corner-case regression。

生产价值是把人工维护的 version folklore 逐步替换成随着测试和部署不断积累的证据。

如果 target identity 碎片化严重到 receipt 永远无法泛化，或者 replay 无法保存 verifier、BTF、privilege 与 attach condition，从而无法复现 outcome，这个方向就失败了。

## 一个保守的 loader policy 不需要因此变慢

主动证据并不意味着每个 process start 都要完整扫描 kernel。

生产实现可以用一个组合 identity 缓存 receipt：kernel package/build identity、BTF digest、相关 config/security-policy identity、privilege profile、BPF artifact hash，以及覆盖 bpftool/libbpf、dependency extractor 和 admission policy 版本的 probe/policy digest。只有所有组件都匹配时才能复用；probe logic 或 policy 变化必须像 target 或 artifact 变化一样让缓存失效。

version metadata 也仍然可以保留成外层 support boundary：

```text
unsupported vendor / known-bad build?
        |
        +-- yes -> reject
        |
        +-- no  -> evaluate artifact requirements
                     |
                     +-- active evidence sufficient -> load
                     |
                     +-- ambiguous edge -> bounded canary
                     |
                     +-- missing capability -> explicit fallback or reject
```

这比两个极端都更可靠。loader 不需要忽略 vendor support policy，也不需要在 kernel 可以直接回答更窄 capability question 的情况下盲信版本号。

fallback 也必须可观测。如果 fentry 不可用而系统选择 tracepoint path，要留下记录。如果 kfunc implementation 不可用而退回 stable helper，也要记录 downgrade。一个 compatibility system 如果静默切换另一条 path，即使 probe 做得再好，事故时仍然解释不出到底跑了什么。

## 什么证据会推翻这个结论？

有三类结果会削弱 artifact-bound capability evidence 的必要性。

第一，大规模实验证明完整 distribution package version 加 vendor release matrix 已经可以几乎零误判地预测 BPF load、attach 和 semantic success。那么 active probing 只会增加运维复杂度。

第二，kernel/libbpf 最终收敛出一个稳定、完整的 capability-query ABI，能够直接描述应用依赖的所有东西，包括 kfunc 与 attach semantics。如果这个接口足够权威而且便宜，自定义 receipt 就可以退化成标准 query 的 signed snapshot。

第三，object-specific verifier/load test 可能已经支配所有更宽泛信号。如果 final artifact 在 final privilege context 下简单 load 一次，就能预测所有重要 compatibility，而且始终可以在 activation 前安全完成，那么额外 general probe 和 semantic canary 价值会很低。

目前证据并不支持这些简化。RHEL 明确写着 version string 不能判断 feature presence，并且发布由 active probing 生成的 BPF capability table。libbpf 本身提供 kernel probe API。CO-RE 使用 target 的真实 BTF，而不是猜布局。kfunc 故意没有 hard stability guarantee，当前 verifier 工作也还在继续调整它们的 argument contract。**所以更强的部署抽象不是“kernel X 足够新”，而是“这份精确 artifact 在这个精确 target、这个 authority 下被 admission 过，并且我们保存了当时的证据与 fallback decision，因此知道自己说的 compatible 到底是什么意思”。**

## 参考资料

- Red Hat. [Managing, monitoring, and updating the kernel: What the version string does not mean](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html-single/managing_monitoring_and_updating_the_kernel/index)，访问于 2026-09-15。
- Red Hat. [Red Hat Enterprise Linux 9.6 Release Notes: Kernel and eBPF facility](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/9.6_release_notes/new-features)，访问于 2026-09-15。
- Red Hat. [RHEL 9.6 Available BPF features](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html-single/9.6_release_notes/index)，访问于 2026-09-15。
- Linux kernel documentation. [libbpf Overview](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html)，访问于 2026-09-15。
- Linux kernel documentation. [BPF Type Format](https://docs.kernel.org/bpf/btf.html)，访问于 2026-09-15。
- Linux kernel documentation. [BPF Design Q&A](https://docs.kernel.org/bpf/bpf_design_QA.html)，访问于 2026-09-15。
- Linux kernel documentation. [BPF Kernel Functions (kfuncs)](https://docs.kernel.org/bpf/kfuncs.html)，访问于 2026-09-15。
- Linux kernel documentation. [HOWTO interact with BPF subsystem](https://docs.kernel.org/bpf/bpf_devel_QA.html)，访问于 2026-09-15。
- libbpf. [`libbpf_probe_bpf_prog_type`, `libbpf_probe_bpf_map_type`, `libbpf_probe_bpf_helper`](https://github.com/libbpf/libbpf/blob/master/src/libbpf.h)，访问于 2026-09-15。
- bpftool. [`bpftool feature` manual](https://manpages.debian.org/unstable/bpftool/bpftool-feature.8.en.html)，访问于 2026-09-15。
- Ubuntu Security. [CVE-2021-3490](https://ubuntu.com/security/CVE-2021-3490)，2026 年更新，访问于 2026-09-15。
- Amery Hung. [PATCH bpf-next v2 00/23: Unify helper and kfunc argument checks](https://lore-kernel.gnuweeb.org/bpf/20260911221956.1F62A1F00893%40smtp.kernel.org/T/)，BPF 邮件列表归档，2026-09-11。