---
date: 2026-08-27
title: "eBPF 安全策略换到不同执行后端后，语义还能保持一致吗？"
description: "eBPF 安全策略可以运行在内核、用户态和 NIC 上。本文提出可执行语义契约、能力覆盖检查与跨后端一致性测试，避免 offload 后悄悄改变安全 verdict。"
tags:
  - Daily Report
  - eBPF
  - Security
  - Networking
  - Offload
research_question: "一个 eBPF 网络安全策略在内核、用户态或 NIC/DPU 后端执行之前，系统需要建立哪些证据，才能确认 allow、drop、redirect、metadata、state 与 failure semantics 没有因为换后端而改变？"
source_cutoff: 2026-08-27
status: daily-report
---

# eBPF 安全策略换到不同执行后端后，语义还能保持一致吗？

假设同一段 XDP 安全程序现在可以跑在三个地方：Linux 内核、用户态 runtime，以及 NIC 或 DPU fast path。三个后端都能接受这段 bytecode，也都能返回熟悉的 XDP verdict。NIC 版本还快很多。

它们执行的还是同一个安全策略吗？

不一定。

一个后端可能能看到 RX hash，另一个看不到。一次 redirect 可能保留自定义 metadata，却丢掉原始硬件 descriptor。某个用户态 runtime 对 helper 的错误行为可能和内核不完全一样。硬件 offload 可能只支持部分 map、metadata、parser depth 或 side effect。常见 packet 走硬件，例外 packet 再送到软件处理时，软件队列一旦过载，系统又需要决定 packet 应该 drop、继续等待，还是走别的路径。

程序可以在指令层兼容，但因为指令周围的执行环境变了，最后做出的安全判断也可能变掉。

<!-- more -->

本文假设 placement 已经决定好了。之前的每日报告 [异构系统里的 eBPF 到底应该运行在哪里？](https://eunomia.dev/zh/research/heterogeneous-ebpf-execution-placement/) 讨论的是：在 kernel、userspace、SmartNIC 和 accelerator 之间，怎样根据 event、state、authority 与 cost 选择执行位置。这里的问题更窄：**安全策略已经被搬过去之后，我们需要什么证据，才能说 enforcement semantics 真的没有变？**

这个区别对安全系统很重要。误 drop 主要伤害 availability；误 allow 则可能直接变成安全漏洞。因此，后端不应该把“不支持这个 field/helper/map/fallback”悄悄解释成“允许通过”。

## 指令兼容只是第一层契约

BPF ISA 现在已经有比较明确的 interoperability 语言。[RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) 定义了 `base32`、`base64`、`atomic32`、`atomic64`、`divmul32`、`divmul64`，以及 historical `packet` 等 conformance group。runtime 和 compiler 可以先明确它们共同支持哪些指令，不再只说一个很模糊的“eBPF compatible”。

但网络安全策略并不只是几条指令。

当前 Linux 的实现已经把这个边界暴露得很明显。内核 [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c) 只允许 `BPF_PROG_TYPE_SCHED_CLS` 和 `BPF_PROG_TYPE_XDP` 进入 device-bound 初始化路径。offload backend 还需要提供 verifier prepare、逐指令检查、finalize、translate 和 map operation。当前 core offloaded-map 分配也只接受 array 和 hash，并不是 Linux 里所有 BPF map 都能直接 offload。

packet metadata 的边界更直接。当前 offload 代码在 fully offloaded program 使用 device-bound metadata kfunc 时会拒绝加载，并给出 `metadata kfuncs can't be offloaded`。也就是说，一段程序可以作为 device-bound kernel program 合法运行，却不能直接变成 hardware-offloaded program。再把第三方 userspace runtime 加进来后，各后端暴露的 metadata surface 只会更不一样。

Linux 的 [XDP RX Metadata 文档](https://docs.kernel.org/networking/xdp-rx-metadata.html) 也明确把 metadata 当成 capability-dependent interface。driver 没实现某个 metadata kfunc 时可以返回 `-EOPNOTSUPP`；当前 frame 没有这项数据时返回 `-ENODATA`。程序把自定义 metadata 放进 `data_meta` 后，producer 与 consumer 还需要额外约定格式。packet 经过 `bpf_redirect_map()` 到另一个设备之后，下一个 consumer 已经拿不到原来的 hardware descriptor，只能看到第一个 XDP program 主动拷贝下来的 custom metadata。

这些不是无关紧要的 implementation detail。如果一个策略需要“drop VLAN X 的 packet”“根据 RX queue/hash 做 rate limit”，或者“redirect 可疑流量同时把 provenance 留给第二阶段”，backend 能看到什么 metadata、redirect 后保留什么信息，本身就是安全语义的一部分。

用户态也一样。[bpftime](https://github.com/eunomia-bpf/bpftime) 并不只是一个 instruction VM。它自己实现 loader、verifier、helper、map、event source 和 attach mechanism，因为光能执行 BPF instruction 并不能复现整个执行环境。它的 XDP support 和 userspace map 也正好提供了一个实际目标：哪些 kernel security contract 可以被 userspace runtime 复现，哪些必须明确声明成 adaptation。

硬件研究从另一个方向得到类似结论。OSDI 2020 的 [hXDP](https://www.usenix.org/conference/osdi20/presentation/brunella) 可以让真实、未修改的 XDP/eBPF program 跑在 FPGA NIC 上，但它不是只做了一个 bytecode decoder。系统还需要 optimizing compiler、专用 eBPF processor，以及 FPGA 版的 XDP map 与 helper。也就是说，要得到 XDP semantics，需要一起重建 XDP 周围的 runtime substrate。

现代 DPU 又增加了一层 split path。当前 [NVIDIA DOCA Flow](https://docs.nvidia.com/doca/sdk/doca-flow/) 文档描述了 fully hardware-accelerated flow pipe，同时允许没有命中 hardware entry 的 packet 被送到 Arm core 做 exception handling，再重新注入硬件。这种结构本身很合理，但安全策略必须明确：exception path 过载、不可用，或者某种 match 只能在 software path 处理时，最终 verdict 由谁负责。

## 现有工作还缺什么

### 1. “后端能跑这段程序”比“后端保持了策略语义”弱很多

loader 可以判断 bytecode 能不能 verify、translate 或 execute，却很少给出这段安全策略依赖的完整环境契约。

缺的是 ISA 上面一层的 **可执行 security semantic contract**。对一个网络策略来说，至少应该声明：

```text
program / hook type
允许读取哪些 packet/context field
依赖哪些 metadata，以及 metadata 缺失时的行为
helper / kfunc 的返回语义
map 类型和 update semantics
state freshness / generation 要求
pass、drop、redirect、modify 等终结 effect
unsupported / unknown 应该怎么处理
```

没有这层契约，backend 可以通过 compatibility test，却在某个关键假设上弱化 enforcement。比如 parser 够不到 encapsulated header、缺失 metadata 被当成 0，或者 redirect 后 provenance 消失，都可能让程序继续合法执行，但结果已经不是原来的 security result。

最直接的实验是 differential test：给 reference 和 candidate backend 放入同样的 policy state、packet 与 context corpus，然后同时比较 verdict、packet mutation、state delta、metadata observation，以及 explicit error state。

### 2. unsupported capability 往往发现得太晚

Linux metadata kfunc 已经把“不支持”(`-EOPNOTSUPP`) 和“这个 frame 没有数据”(`-ENODATA`) 分开。这个区分很有用，因为 policy 可以为两种情况定义不同处理方式。跨 backend deployment 也需要在真正接流量之前做类似判断。

缺的是 **coverage-aware activation gate**。backend 应该暴露 compiled policy 真正需要的 capability；只要安全相关 requirement 缺一个，就不能直接激活，除非 policy 明确声明了安全 fallback。

否则就会出现危险的模糊状态：硬件 path 少支持一个 match 或 state operation，系统直到 packet 到来时才发现，然后这个 fallback 可能变成隐藏的 fail-open path，或者变成没有上界的 exception queue。

实验可以故意一次删掉一个 required capability，确认 policy 要么拒绝 activation，要么进入预先声明的 exception path。任何 mutation 都不能把 `unknown` 悄悄转换成 `allow`。

### 3. 后端差异经常藏在 state 和 side effect，而不只是 return code

两个 backend 对一个 packet 都返回 `XDP_DROP`，并不代表状态也相同。它们可能更新了不同的 counter、conn state、rate-limit bucket 或 policy generation。这种差异通常要等到下一个 packet 才会暴露。

缺的是对 **observable state transition** 的 conformance model，而不是只比较当前 verdict。安全测试要比较那些会跨 event 保存、并影响后续决策的 policy-relevant state。

否则会出现 delayed semantic drift：第一包完全一致，但 map atomicity、eviction、update ordering 或 freshness boundary 不同，后续 packet 才开始分叉。

因此 benchmark 应该是 sequence，而不是一个 packet。需要 replay flow，同时加入 state mutation、concurrent update、resource pressure 与 backend reset，并在每一步比较 verdict 和 reachable policy state。

### 4. exception path 在过载时也必须有安全语义

hardware pipeline 把常见 case 加速，把 miss 或 unsupported case 交给 software，通常是正确架构。真正的问题是 software exception path 慢了、挂了或者队列满了以后怎么办。

缺的是明确规则：**当 authoritative exception path 暂时无法回答时，fast path 允许做什么？** 不同 policy 的答案可能是 fail closed、使用很短的 lease、rate-limit、quarantine，或者把 explicit unknown 交给下一个 enforcement layer。

如果这个规则没有写进 contract，overload 就可能变成 policy bypass。只测 steady-state throughput 的 benchmark 看不到它。

实验应该主动压满或者停止 exception processor，同时持续生成必须走 exception path 的流量。重点测 false allow、false deny、queue growth、recovery，以及系统有没有把 degraded state 错报成“策略已经完整生效”。

## 有学术价值和生产价值的方向

### 1. 编译 security-semantics contract，并做 differential conformance

**Gap.** ISA conformance 和 successful loading 都不能证明不同 execution environment 会做出相同的 network-security decision。

**Mechanism.** 在 policy build artifact 旁边生成一份 machine-readable contract。contract 从程序、attach configuration 与 policy compiler 中提取 required hook、context field、metadata、helper/kfunc behavior、map operation、persistent-state invariant、允许的 terminal effect 与 failure policy。

然后测试 harness 用同一个有边界的 corpus 跑 reference Linux 和各个 target backend。Linux 已经提供一个很有用的 primitive：[`BPF_PROG_RUN`](https://docs.kernel.org/bpf/bpf_prog_run.html) 可以让 XDP 等多类 BPF program 对 userspace 提供的数据和 context 执行，并把 result 返回给 userspace。普通 test-run mode 会刻意屏蔽真实 packet side effect，所以它不能成为完整 oracle，但很适合作为纯 packet/context 行为的 repeatable reference。涉及 side effect 的 case 再用 controlled live execution，并检查 contract 声明的 state delta。

比较内容不能只有 `retval == retval`：

```text
input packet + context + policy generation
        |
        +-- reference kernel ------> verdict, mutation, state delta, evidence
        |
        +-- userspace backend -----> verdict, mutation, state delta, evidence
        |
        +-- NIC/DPU backend -------> verdict, mutation, state delta, evidence
```

每个 mismatch 都必须分类：equivalent、explicitly unsupported、按明确规则 intentional adaptation，或者错误。不能再用“程序跑起来了”结束判断。

**和相关工作的区别。** RFC 9669 给的是 instruction-level conformance group。hXDP 证明 FPGA target 可以重建 XDP map/helper。这里进一步问：哪些 environment assumption 真正参与了 security decision，以及 backend 能否针对这些 assumption 给出有边界的 behavioral conformance evidence。

**Artifact.** 一套 contract schema、libbpf-side extractor/compiler、Linux/bpftime-like userspace/NIC-DPU adapter，以及可重复使用的 packet/state corpus。

**Evaluation.** 选择 XDP firewall、redirect、rate-limit、load-balancing 和 stateful allow-list。系统性改变 metadata availability、map behavior、parser depth、redirection、state generation 与 concurrent update。primary metric 是相对 reference policy 的 false allow；secondary metric 包括 false deny、explicit unsupported/unknown、state divergence、test coverage、activation latency、steady-state cycles、throughput 和 memory。

**Academic value.** 把 ISA interoperability 与 environment-dependent security semantics 拆开，给 heterogeneous BPF runtime/accelerator 一个可以独立研究的 equivalence target。

**Production value.** deployment pipeline 可以在接流量之前拒绝不满足要求的 backend，而不是在线上才发现语义不兼容。

**Failure condition.** 如果 real policy compiler 无法自动提取有用 contract，最后仍需要人工把整段程序再描述一遍；或者 reference 太依赖某个 backend，根本无法定义 portable semantics，那么这套方法就容易退化成 documentation，而不是 executable assurance。

### 2. activation 必须知道能力覆盖范围，失败必须显式

**Gap.** backend 即使实现了 95% 的 policy，也可能因为剩下 5% 刚好是决定 allow/drop 的条件而变得不安全。

**Mechanism.** 每个 backend 发布带版本的 capability manifest，并明确每种 capability 的 failure behavior。activation 时把 policy contract 和 manifest 做匹配。结果不是一个“支持率”，而是 required semantics 的集合关系：

```text
required(policy) ⊆ provided(backend)
```

关系不成立时，loader 要么拒绝 deployment，要么安装一个已经声明过的 split path。split path 给 packet 携带一个很小的 witness，说明为什么离开 accelerated path，以及最终 decision authority 在哪里。unsupported 和 overloaded 都要保留成 observable state，不能默认为 `PASS`。

manifest 还需要区分稳定 capability 与 frame-specific absence。Linux 的 `-EOPNOTSUPP` 和 `-ENODATA` 就是很好的先例。parser/tunnel depth、map type 或容量限制、redirect 时 metadata preservation、exception path dependency 也都应该进入 contract。

**和相关工作的区别。** device feature discovery 已经存在于很多 subsystem，DOCA 这类 pipeline 也已经有 hardware/software exception path。这里的变化是把 backend capability 和 compiled security-policy requirement 绑定起来，让 activation 与 fallback 自己成为 enforcement semantics 的一部分。

**Artifact.** backend capability schema、policy-to-capability matcher、fail-closed admission gate，以及 split execution 的 per-packet exception witness。

**Evaluation.** 一次改变一个 capability：删掉 RX hash/timestamp support、限制 parser depth、关闭 required map operation、压满 exception queue、reset device 或断掉 software handler。比较 silent fallback、reject-on-mismatch 与 witness-carrying split execution。先测 false allow，再测 availability loss、exception volume、offload coverage 和性能。

**Academic value.** 把 partial acceleration 从 implementation detail 变成 policy compiler 和 execution substrate 之间可组合的 security contract。

**Production value.** operator 在 activation 前就能知道“hardware offload enabled”到底表示完整 enforcement、带明确 fallback 的部分 enforcement，还是根本不能安全部署。

**Failure condition.** 如果 capability manifest 变化太快，或者根本无法用合理粒度描述 semantic limit，保守 admission 可能频繁关闭 acceleration。那时单一 backend 的简单系统反而更合适。

### 3. 做一个专门攻击 backend assumption 的 semantic-mutation benchmark

**Gap.** 现在很多 performance evaluation 能证明 offload/userspace implementation 很快，却不会主动改变那些最容易影响安全 verdict 的环境假设。

**Mechanism.** 构造一套每个 case 只 mutation 一个 environment assumption 的 benchmark，同时给出已知 ground-truth policy result。例如：

- RX hash、timestamp、VLAN、queue、custom metadata 存在或缺失；
- redirect 前后有没有主动复制 provenance metadata；
- tunnel/header depth 刚好在 parser capability 内外；
- map update race、atomic operation、capacity pressure 与 generation change；
- exception handler delay、queue overflow、crash 与 recovery；
- policy state 保留或重建时的 userspace/runtime restart；
- long-lived stateful flow 中途发生 device reset 或 backend switch。

metamorphic test 特别适合这里：如果 policy 根本不依赖 field X，删掉 X 不应该改变 verdict；如果 policy 依赖 X，那么 target 必须保留 X，或者按照 contract 明确报告 unsupported/unknown。

**和相关工作的区别。** 常见 packet-processing benchmark 重点是 throughput、latency 或 instruction execution。这里把 semantic mutation 设为 independent variable，把 false security decision 设成主要 outcome。

**Artifact.** 开放的 packet/context trace、state snapshot、backend fault adapter、expected verdict，以及区分 false allow、false deny、unknown 与 state divergence 的统一报告格式。

**Evaluation.** 在 kernel-native XDP、bpftime 这类 userspace eBPF runtime，以及至少一个 hardware/DPU backend 上跑同一个 corpus。比较三种 deployment policy：只检查 ISA/loadability、只做 differential conformance、以及 conformance + coverage-aware activation。性能比较时固定 CPU/offload budget。

**Academic value.** 给“policy 可以跨 heterogeneous execution substrate portable”这种 claim 一个可以被证伪的 benchmark target。

**Production value.** 它可以直接变成 backend upgrade、driver change、新 offload target 与 policy compiler release 的 regression suite。

**Failure condition.** 如果 corpus 无法表示 proprietary hardware semantics 或真实生产 state，它可能只能验证一个比较窄的 common subset。即便如此，把这个 subset 准确说清楚也比宣称 universal compatibility 更有价值。

## 什么证据会改变这个结论？

最强的反例是：现有 loader 与 hardware toolchain 已经会拒绝所有 security-relevant unsupported operation，而且现实中被 offload 的 policy 只使用一小组非常稳定的 common subset。如果广泛 differential test 显示 kernel、userspace 和 hardware target 在 metadata loss、state pressure、redirect、exception overload、backend reset 下都能保持相同 verdict 与 policy-state transition，那么再加一层 semantic contract 的收益可能不大。

第二个反例是成本和复杂度。如果 capability witness 或 cross-check 必须每个 packet 都做一次，就会抵消 offload 的意义。因此本文更偏向把大部分 assurance 放在 build/activation time，只把真正需要的 state/exception evidence 留在 hot path。

所以结论并不是“每个 backend 都必须完整模拟 Linux BPF”。更窄也更可验证的说法是：**安全策略只能在它真正依赖的 environment semantics 都被保留的 backend 集合中 portable；任何不支持的语义都应该保持显式，而不是被静默翻译成另一个 verdict。**

这比 bytecode compatibility 强，但比 general heterogeneous placement 窄。它给 kernel、userspace、NIC 与 DPU implementation 一个可以先测安全一致性、再谈性能收益的边界。

## Sources

- IETF: [RFC 9669, BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html)
- Linux kernel source: [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c)
- Linux kernel documentation: [XDP RX Metadata](https://docs.kernel.org/networking/xdp-rx-metadata.html)
- Linux kernel documentation: [Running BPF programs from userspace (`BPF_PROG_RUN`)](https://docs.kernel.org/bpf/bpf_prog_run.html)
- Brunella et al., OSDI 2020: [hXDP: Efficient Software Packet Processing on FPGA NICs](https://www.usenix.org/conference/osdi20/presentation/brunella)
- NVIDIA: [DOCA Flow](https://docs.nvidia.com/doca/sdk/doca-flow/)
- Eunomia: [bpftime userspace eBPF runtime](https://github.com/eunomia-bpf/bpftime)
