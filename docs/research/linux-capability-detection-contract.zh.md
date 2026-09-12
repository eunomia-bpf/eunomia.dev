---
date: 2026-09-12
slug: linux-capability-detection-contract
title: "为什么 Linux 能力检测不能只看内核版本？"
description: "Linux 能力检测不能只依赖内核版本：发行版回移植、启动配置和语义修复会让版本号与真实能力分离。本文比较运行时协商机制，并提出可追溯的能力证据回执。"
tags:
  - Daily Report
  - Linux
  - Compatibility
  - System Calls
  - Runtime
  - Portability
research_question: "当发行版回移植、启动配置、运行时策略和语义修复都可能偏离内核版本号时，可移植的 Linux 软件应该怎样判断一个功能在当前环境里是否真的可用？"
source_cutoff: 2026-09-12
status: daily-report
---

# 为什么 Linux 能力检测不能只看内核版本？

假设一个服务启动后看到机器报告 Linux 5.14。它想使用某个安全机制、较新的 `io_uring` 操作，或者依赖 `openat2()` 的一种行为。最简单的部署逻辑是查一下“这个功能从 Linux X 开始支持”，再拿 X 和 `uname -r` 比较，决定走新路径还是 fallback。

这种写法很省事，因为它把兼容性问题变成了一个版本号比较。但真实的 Linux 部署并不保持这种一一对应关系。发行版会把较新的子系统改动回移植到较老的 upstream 基线；某个功能可能编进了内核，却在启动时被关闭；沙箱或兼容层也可能让进程看不到宿主机已经实现的 syscall。即使 API 本身存在，后续的语义修复也可能改变应用真正依赖的边界行为，而应用最初检查的 ABI 版本并没有变。

**所以，Linux 能力检测真正需要的是当前运行接口给出的证据，而不只是从内核版本推断出来的猜测。更难的一步，是保存足够的证据，说明究竟证明了哪种行为、这个结论在哪个执行环境中成立，以及程序为什么选择了某条路径。**

<!-- more -->

这篇报告是当前 eBPF 路线中的一次相邻系统 detour。最近十篇 Daily Report 已经达到配置允许的 **7 篇 eBPF-centered** 上限，今天如果继续发布 eBPF 主题就会破坏滚动比例。这个兼容性问题仍然与 [bpftime](https://github.com/eunomia-bpf/bpftime) 这类可移植 runtime 有关，但本文讨论的是一般 Linux userspace interface，而不是 BPF 专用的能力协商。

## Linux 能力检测本来就是多种不同协商契约的组合

反对“只看版本号”的最好证据，其实来自 Linux 接口本身：不同子系统已经各自设计了多种兼容性协商方式。

Red Hat 的文档直接解释了为什么 release string 不能完整描述能力。RHEL 9 的内核可以继续报告 5.14 这一 upstream 基线，同时某些子系统已经包含来自远更新 upstream kernel 的改动。版本号能告诉你这是哪个发行版内核系列，却不能列出里面究竟回移植了哪些功能。

Landlock 更明确。它要求 userspace 通过 `landlock_create_ruleset(..., LANDLOCK_CREATE_RULESET_VERSION)` 查询正在运行的 Landlock ABI，再只启用该 ABI 支持的 access rights，而不是根据 kernel version 猜测。2026 年 8 月的当前内核文档又增加了一个很有意思的层次：`LANDLOCK_CREATE_RULESET_ERRATA` 可以返回语义修复的 bitmask。也就是说，一个功能可以“已经存在”，粗粒度 ABI 也相同，但某个应用关心的边界行为仍可能因为修复是否存在而不同。

`openat2()` 采用另一种办法。它的 `struct open_how` 明确允许未来追加字段，传入的结构体大小本身就是隐式版本。如果 userspace 传入了当前 kernel 不认识的扩展字段，而且这些未知字段非零，内核会返回 `E2BIG`。man page 甚至说明了如何通过 size 探测当前内核理解到什么结构版本。这里的兼容性不是一个显式 ABI 数字，而是由结构形状协商出来的。

`io_uring` 又形成了另一套机制。`IORING_REGISTER_PROBE` 可以查询支持哪些 opcode，`io_uring_setup()` 的 feature bits 描述部分运行行为。当前 man-pages 还记录了从 kernel 6.15 开始提供的 `IORING_REGISTER_QUERY`：它不需要先创建 ring，就能查询 opcode、flag 和子系统相关能力。子系统逐渐走向更丰富的 runtime query，本身就说明一个 release number 很难方便地表达真实能力矩阵。

这些设计并不是互相矛盾的错误。每一种机制面对的兼容性问题不同。真正麻烦的是，可移植 userspace 最终拿到的证据散落在 ABI version、bitmask、opcode probe、可扩展结构、启动配置、返回码和执行策略里。

## “这个功能支持吗”实际上混在了一起的四个问题

第一个问题是**实现是否存在**。当前 kernel 或兼容层到底有没有实现这个 syscall、opcode、flag 或数据结构？`ENOSYS`、opcode probe 或专门的 query interface 通常可以回答。

第二个问题是**当前运行环境是否可用**。Landlock 可以已经编译进 kernel，却因为启动配置没有启用。某些 `io_uring` 操作还会受到 privilege 或 ring mode 的限制。源码里有这个功能，不等于这个进程现在就能用。

第三个问题是**语义处在哪个层级**。一个 ABI number 或 opcode bit 可能只能说明粗粒度 generation，而后续 bug fix 会改变应用在意的 edge case。Landlock 新的 errata query 把这件事直接暴露出来：有时 userspace 需要知道的不只是“接口存在”，而是“某个语义修复是否已经在这台机器上生效”。

第四个问题是**当前 execution context 是否有权使用**。container、seccomp policy、virtualized kernel、LSM 配置、namespace 和 credential 都可能让两条运行在同一个名义 kernel 上的进程看到不同的 usable capability set。因此，一个 host-global cache 很可能替 sandbox 里的进程回答了错误的问题。

版本号比较把这四个问题压成了一个 proxy。直接 probe 已经好很多，但如果软件不记录这个 probe 到底证明了什么、哪个决策依赖它，后续仍然很难解释行为。

## 现有研究还缺什么

第一，**不同 Linux API 之间缺少统一的证据模型**。Landlock 能给 ABI 和 errata mask，`io_uring` 能给 opcode 与 flag support，`openat2()` 能协商结构体大小，但应用通常只是把这些结果变成 loader 代码里的几个临时 boolean。出问题以后，很难判断 fallback 到底是因为 kernel 没这个功能、功能被关闭、进程被策略拒绝，还是应用根本不认识较新的接口。

第二，**feature presence 之上的语义可信度没有统一表达**。能力 API 一般只能回答本子系统定义的问题，却很少直接表达应用真正需要的高层保证，比如“在这些 flag 下，路径解析不会逃出指定 root”或者“这个 sandbox 对我依赖的网络行为处理正确”。Landlock 的 errata 机制说明，ABI 出现之后，行为级差异仍然可能值得单独表达。

第三，**能力证据的作用域和寿命往往不清楚**。有些观察结果在整个 boot 内都稳定，有些取决于进程 credential、namespace、seccomp state、container runtime 或虚拟 syscall surface。很多 capability cache 不声明自己的 validity scope，把 host 外部的 probe 结果直接复用到更受限的执行环境，就会产生 stale-evidence bug。

第四，**fleet compatibility 缺少可复现的反例测试**。发行版回移植本来就会打破 upstream version 与 feature set 的简单映射，但很多 CI matrix 仍主要用 kernel release 标记机器。这样一旦失败，很难分清到底是版本 lineage、具体 backport、配置差异、语义修复还是 policy layer 导致的。

## 兼具学术价值和生产价值的方向

### 1. 把 capability probe 变成有类型的证据回执

缺的不是 probe。Linux 已经有很多不错的 probe。缺的是一个共同格式，能记录一次 probe 成功或失败到底证明了什么。

一份很小的 capability receipt 可以是：

```text
requirement = landlock.net.bind-tcp
probe = LANDLOCK_CREATE_RULESET_VERSION
result = abi>=4
semantic_fixes = {erratum-1, erratum-2}
kernel_build = package + build-id
execution_scope = boot + userns + seccomp-profile + credentials
observed_at = process-start
chosen_path = landlock-network-policy-v2
fallback = filesystem-only-policy
```

机制可以是一套 userspace library 加 schema，由不同 adapter 把本地子系统的 probe 结果填进去。它不应该替代 Landlock、`io_uring` 或 `openat2()` 自己的协商方式，而是保留这些 source-native semantics，再把证据和真正消费它的 application decision 绑定起来。

最强 baseline 是现在常见的手写 feature detection。可以在 mainline kernel、长期维护的发行版 kernel、不同 boot config、container 和 syscall virtualization 环境中比较两种方案，并注入 backport、feature disabled、syscall denied 与部分 semantic errata。指标包括误启用、无必要 fallback、解释一次决策所需时间、receipt 大小和启动开销。

学术价值在于建立一种 capability evidence model，把 presence、availability、semantics 和 authorization 分开，而不是把“支持”压成一个 bit。生产上，runtime、database、storage engine、sandbox 或 agent executor 都可以在 startup 和 backend-selection 边界接入这套机制，替代散落的 kernel-version conditional。

如果普通 per-API probe 加少量日志已经能以更低复杂度达到相同的决策准确率和事后可解释性，这个方向就不值得做。通用 schema 只有在它真的捕获了多个 API 反复出现的结构时才有价值。

### 2. 为版本号兼容规则建立专门的 counterexample corpus

可以故意构造一组 release-number heuristic 会答错的场景：一台机器在较老 upstream base 上带发行版 backport；另一台 kernel 更新，但功能被关闭；container 拒绝 syscall；virtualized kernel 根本没实现它；两个环境暴露相同粗粒度 ABI，却在一个已记录的 erratum 上不同。

artifact 不应是“kernel X 支持 feature Y”的静态表，而应该是一组很小的 requirement-specific probe 和行为测试。每个 case 同时记录版本 heuristic 的预测、native probe 的结果，以及真正执行一次应用依赖的 semantic property 后得到的结果。

评测可以选择若干真实 Linux library 或 runtime，比较它们当前使用的 version gate、compile-time macro 和 direct probe。主要指标不是测试覆盖率，而是 **capability misprediction**：误启用不可用或语义不满足的路径，以及把其实可用的功能错误关闭。再通过 ablation 去掉 backport 或 policy layer，看每类偏差贡献多少。

学术贡献是一套可以量化 mixed Linux fleet 兼容性失败的 taxonomy。生产团队则可以在扩大 rollout 之前，让这套 corpus 对自己真正发布的 kernel image 和 sandbox profile 运行一次。

如果代表性的生产 kernel 和运行环境几乎从不与简单版本规则发生冲突，或者 native probe 已经能抓住全部有意义的反例，这个方向应该放弃。目标是找到真实 divergence，而不是人为制造边角故障。

### 3. 给 capability evidence 明确的有效期和作用域

一次 capability observation 应该说明它在哪里、多久有效。kernel-build property 可能整个 boot 都不变；boot-time LSM setting 在重启后就需要失效；seccomp 或 namespace 相关结果可能只对一个 process lineage 成立；sandbox 外观察到的能力，也不能自动授权 sandbox 内继续使用。

机制可以很简单：给每份 receipt 附一个 scope key，例如 kernel build identity、boot ID、namespace identity、sandbox-policy digest 和 credential class。只有 requirement 声明相关的 component 没变化时才允许复用 cache，否则重新 probe 或走保守 fallback。

评测比较 host-global cache、每次使用都 probe、以及 lease-scoped cache。在 container start、sandbox policy 变化、privilege drop、kernel upgrade 和 restart 中测 stale-positive decision、重复 probe 数、启动延迟和保守 fallback 成本。一个很直接的 adversarial case 是：host-level probe 成功后，再把工作进程放进更严格的执行环境。

学术问题是：面对异构 OS capability，怎样推导最小 validity scope，同时避免把每次能力检查都退化成一次性 probe。生产价值主要在长期运行的 runtime 和 fleet agent，它们会缓存 backend choice，但之后又在不同 policy 下创建 worker。

如果应用关心的所有 capability 在整个进程生命周期都不可变，而且 process-start probing 本来就足够便宜，那么 lease 机制没有必要。那种情况下额外状态只会增加复杂度。

## 给可移植 Linux 软件的一条实际规则

最简单可用的策略是分层处理。

把 kernel release 当作**方向信息**，而不是证明。它适合日志、粗粒度支持策略，以及判断哪些 probe 值得尝试。

把子系统自己的协商接口当作**能力证据**。优先使用 Landlock ABI query、`io_uring` probe/query、可扩展结构协商或者文档定义的 syscall 行为，而不是猜一个 minimum kernel version。

如果这个功能关系到安全或 correctness，再验证应用真正依赖的**语义性质**。一次 syscall 成功，不自动等于它证明了应用期待的全部保证。

最后，把结果和它的**执行作用域以及 fallback decision** 绑在一起。以后 operator 问“为什么这个 worker 走了慢路径，另一个却没有”时，系统应该拿出证据，而不是重新从 `if (kernel_version >= ...)` 里猜答案。

这和此前的[特定架构 eBPF 可移植性报告](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)有关，但边界不同。那篇文章问的是一个 portable BPF semantic artifact 如何选择架构相关实现；本文讨论的是普通 Linux userspace software 如何面对互不相同的 kernel API 与发行版 backport。对后续 eBPF compatibility 系列来说，这意味着应该聚焦 verifier、CO-RE、kfunc、attach 和 persistent-state 这些 BPF 特有语义，而不是重复一遍通用 Linux capability receipt。

## 哪些结果会改变这个判断？

如果发行版 kernel version 在实践中已经足够可靠地标识 capability，这个结论会明显变弱。可以做一个大规模 fleet study：收集常见 RHEL、Ubuntu、Debian、cloud、mainline 与 virtualized kernel，只根据 release string 预测可用功能，再与 native probe 和 semantic test 比较。如果 false positive 与 false negative 都接近零，多数应用就没有必要承担更复杂的直接能力证据机制。

如果 Linux 以后收敛出一种统一的 machine-readable negotiation contract，可以同时表达 presence、configuration、semantic revision 和 execution-policy constraint，本文提出的跨子系统 receipt 也会失去价值。`IORING_REGISTER_QUERY` 以及 Landlock 的 ABI/errata query 已经在各自子系统内朝这个方向走，但今天还没有统一 contract。

最后，并不是每个 feature 都值得做 semantic test 或持久保存 receipt。一个有安全 fallback 的 best-effort 性能优化，完全可以先试操作，失败就回退。只有当错误的 positive 会破坏安全或 correctness、fleet 决策很难回滚，或者 operator 必须复现两台看起来相似的 Linux host 为什么行为不同，更强的证据机制才值得付出成本。

所以实际标准不是“永远不要看 `uname`”，而是：**用版本号判断大致范围，用运行时接口证明真实能力，再保存足够且带作用域的证据，解释为什么 backport、配置或语义差异会让两个看起来相似的 kernel 做出不同的行为。**

## References

- Linux kernel documentation, [Landlock: unprivileged access control](https://kernel.org/doc/html/latest/userspace-api/landlock.html), including ABI and errata negotiation, accessed 2026-09-12.
- Linux man-pages, [`io_uring_register(2)`](https://man7.org/linux/man-pages/man2/io_uring_register.2.html), including `IORING_REGISTER_PROBE` and `IORING_REGISTER_QUERY`, accessed 2026-09-12.
- Linux man-pages, [`openat2(2)`](https://man7.org/linux/man-pages/man2/openat2.2.html), including extensible-structure negotiation, accessed 2026-09-12.
- Red Hat Enterprise Linux documentation, [Managing, monitoring, and updating the kernel](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html/managing_monitoring_and_updating_the_kernel/), including the RHEL backport model, accessed 2026-09-12.
- Linux kernel documentation, [Linux ABI description](https://kernel.org/doc/html/latest/admin-guide/abi.html), accessed 2026-09-12.
