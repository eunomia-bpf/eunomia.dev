---
date: 2026-09-08
title: "eBPF 把操作交给原生代码后，怎样控制可信边界？"
description: "eBPF 原生操作可以保留已验证语义，却不应把每个优化器都纳入可信计算基。本文提出独立证书、effect 边界与故障评测。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - 程序验证
  - 系统安全
  - 编译器
research_question: "eBPF 怎样把已经验证的语义操作委托给原生代码，同时避免把每个优化器、后端和原生实现都纳入可信计算基？"
source_cutoff: 2026-09-08
status: daily-report
---

# eBPF 把操作交给原生代码后，怎样控制可信边界？

假设一个 eBPF 优化器发现，原本需要多条 BPF 指令完成的操作，在某种 CPU 上可以用一条原生指令实现。原来的 BPF 序列已经通过 verifier，目标机器支持这个快路径，loader 也正确选中了实现。此时还差一个问题：**谁来证明这段 native code 只做了原始 BPF 程序被允许做的事情？**

verifier 分析的是 BPF 指令，以及它知道语义的调用边界；优化器之后塞进去的任意机器码并不自动处于这套证明里。一旦 specialization 可以用 native code 替换 verifier 看过的逻辑，native implementation 里的一个错误就可能直接扩大成 kernel safety 问题，即使原始 BPF 程序完全安全。

本文的判断是：**原生委托应该同时携带 verifier 可见的 semantic witness，以及可以独立检查的 implementation certificate。新增一个优化，不应该自动把整个 optimizer 和 code generator 一起放进可信计算基。**

<!-- more -->

上一篇关于[特定架构 eBPF 优化可移植性](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)的报告讨论：某个 native implementation 能不能在当前机器上使用，不能用时如何确定地退回 portable BPF。那是 capability 与 portability 问题。本文假设实现已经被正确选中，只追问它为什么有资格获得与原始 BPF 相同的执行权限。

更早一篇关于[运行时画像驱动 eBPF specialization](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/)的报告讨论 workload assumption 会不会过期。本文的错误不需要等 workload 变化。一段实现错误的 native emit 第一次执行就可能偏离原始语义。

## Verifier 可以约束调用边界，却不会自动证明被调用函数

Linux 的 BPF kernel function，也就是 kfunc，是一个现成的边界例子。当前 Linux 文档明确体现出两个层次。

第一层是 verifier 可以检查很丰富的调用契约。BTF 类型限制参数，而 `KF_ACQUIRE`、`KF_RELEASE`、`KF_RET_NULL`、`KF_SLEEPABLE`、`KF_RCU`、`KF_DESTRUCTIVE` 等标记描述资源生命周期、空值、执行上下文与危险 effect。例如 `KF_ACQUIRE` 会让 verifier 继续证明返回的引用最终被释放，或者转移到支持的 map 状态。

第二层是边界后面的实现仍然是普通 kernel code。Linux 文档说明已有内核函数可以直接注册成 kfunc，但维护者仍需审查 BPF 会在什么上下文调用它，以及这样做是否安全。verifier 检查声明出来的边界；它不会证明函数体实现正确。

这种划分本身很合理，否则 verifier 为了验证一个 BPF program 就需要理解整个 Linux。但如果 userspace optimizer 或第三方模块可以不断加入 machine-specific operation，它也把 trust 问题暴露得很清楚：call-site contract 只有在被委托实现确实遵守契约时才有意义。

通用的 [eBPF verifier 文档](https://www.kernel.org/doc/html/latest/bpf/verifier.html)也是同样的边界。verifier 对 BPF 指令做符号执行，约束函数调用的参数与返回状态。这是一条 interface-level proof boundary，不是对后续被替换进去的任意机器码做等价证明。

## Kops 把新增的可信代码明确暴露出来

[Kops](https://arxiv.org/abs/2606.24213) 直接把这笔 trade-off 写得很清楚。一个 Kops operation 有两种表示：普通 eBPF 指令组成的 proof sequence，以及架构相关 JIT 使用的 native emit。proof sequence 仍然是 verifier 可见语义；native emit 则必须实现相同语义。

Kops 对文中的 hardware idiom 使用 Lean 4 证明 native emit 与 BPF proof sequence 等价。论文报告 microbenchmark 最高 24% 加速，生产应用最高 12%；它还演示 whole-program native replacement，在评测中达到 2.358x，但代价是更大的 trusted computing base（TCB）。

因此 native delegation 不只有“快不快”一个维度。保留多少 verifier-visible semantics、又把多少 native code 放进必须信任的范围，会直接改变安全成本。

## Jitterbug 说明 deployed JIT 也不能靠成熟度推断正确

更早的 Jitterbug 工作给出一个直接反例。它为 BPF JIT 定义精确 correctness specification，验证了一个新的 RV32 JIT，并在五个已经部署的 Linux BPF JIT 中发现并修复 16 个此前未知的问题。

Jitterbug 的核心性质是 source BPF semantics 与 target instruction sequence 的行为等价。这正是 extensible native-operation interface 需要的性质。它同时提醒我们：verification framework 自己也有边界，哪些状态被建模、哪些 translation/encoding step 被验证，都决定真正的 TCB 在哪里。

对 Kops 这类细粒度 operation 来说，这反而可能是机会。一个只替换小型 arithmetic idiom 的 native implementation，比完整 JIT 更有机会携带紧凑 proof 或 translation-validation artifact，而 checker 本身可以远小于 optimizer 的搜索与代码生成逻辑。

## 身份与完整性不等于语义等价

当前 Linux 的 [BPF signing 文档](https://www.kernel.org/doc/html/latest/bpf/signing.html)提供另一条清楚的边界：有效签名可以证明 bytecode 与被覆盖 metadata 来自预期 producer、没有被篡改；文档也明确指出 signing 与 capability check、verifier 是正交的。

native-operation package 同样应该绑定 exact machine-code artifact 的 hash 或签名，但这仍然不够。它可以证明**到底跑的是哪份实现**，却不能证明这些 bit 与 verifier-approved semantics 等价。trust contract 至少需要两种 identity：具体 artifact identity，以及它获得 semantic authority 的证据。

## 真正缺少的是 semantic authority 与 implementation authority 之间的契约

这里最好把两个经常混在一起的问题分开。

**Semantic authority** 定义 operation 被允许做什么。对 native BPF operation 来说，可以由普通 BPF proof sequence 加 explicit effect summary 表示：哪些 register 是输入输出，可以读写哪些 context 或 map memory，是否能调用 helper/kfunc，是否可以 sleep、allocate、取得 reference 或产生外部 effect。

**Implementation authority** 回答为什么这一段具体 machine code 可以代替上面的语义操作。它应该把 exact implementation hash 与 semantic witness 绑定，再附上能由独立 checker 验证的证据，而不是只相信生成它的 optimizer。

一个最小 package 可能类似：

```text
semantic_op = rotate64_v1
proof_seq_hash = sha256(portable_bpf_sequence)
effects = {read: r1,r2; write: r0; memory: none; calls: none}

native_impl_hash = sha256(machine_code)
target = x86_64
certificate = proof-or-translation-validation-result
checker_version = v3
```

具体字段不是本文要提前冻结的 ABI。关键是职责分离：userspace optimizer 可以大胆搜索，backend 可以生成机器码，一个更小的 checker 决定产物能否获得权限。如果 optimizer 出错，应该由 checker 拒绝实现，而不是把 optimizer 本身变成 kernel safety 的一部分。

这与上一篇的 capability manifest 不同。capability negotiation 回答“这个 implementation 能不能在这里跑”；trust contract 回答“为什么它可以获得与 verifier 已批准语义相同的权限”。

## 现有研究还缺什么

### kfunc 式契约能约束边界，却不能证明实现等价

Linux kfunc metadata 已经能表达 pointer trust、ownership、RCU、sleepability 和 destructive behavior，因此 verifier 可以拒绝很多不安全调用。但边界之后仍依赖 kernel review 与 implementation correctness。

对于少量、稳定、in-tree 的 kernel function，这可能是合理工程取舍。如果 userspace optimizer 或第三方 module 可以加入大量 machine-specific operation，review-only trust 则会随着 operation 与 backend 数量一起增长。

真正缺的是故障证据：保持 kfunc-like signature 与 declared effect 不变，只修改 native body，看看 compact semantic/effect contract 加独立 checker 能捕获多少 call-site typing 捕获不了的错误。

### proof-linked operation 仍然需要可部署的 checker boundary

Kops 说明 native emit 可以和 verifier-visible proof sequence 建立形式关系；Jitterbug 说明 target-code equivalence 确实能发现真实 JIT bug。但 production extension 仍必须决定哪些东西在 build time、load time、kernel integration time 检查，哪些最终仍然被信任。

如果 certificate 只是因为“某个 compiler pipeline 生成了它”就被相信，那么 TCB 只是换了位置。如果一个小 proof checker 或 translation validator 能独立拒绝坏 implementation，generator 才真的可以留在 TCB 外面。

应该测的不是又一次 speedup，而是 operation library 与 architecture matrix 扩大时，trusted-code growth 和 escaped semantic bug 怎样变化。

### whole-program native replacement 会扩大一次错误的 failure radius

把一个算术 idiom 换成一条机器指令，与把整个 BPF program 换成 native code，都能叫 native specialization，但失败半径完全不同。小 operation 可能不碰内存；完整程序则可能读写 packet/context、调用 helper、更新 map，并经过复杂控制流。

所以 trust model 不能只有一个“equivalent”标签，还需要 effect envelope。如果完整证明还做不到，系统至少应该知道 native body 获准使用哪些 effect，并在越界时 fail closed。

### provenance 能定位坏实现，却不能阻止坏实现

artifact hash、signature、optimizer generation 与 JIT dump 对事故分析很有价值，但它们不能阻止一份“身份正确、语义错误”的实现被执行。因此 debugging provenance 与 trust contract 是互补关系，不是替代关系。

这也把下一篇可能的研究问题留得很干净：本文只讨论 native code 的 admission authority；后续可以单独研究 operation 通过以后，operator 怎样重建真正执行过的 specialization decision 与 machine-code artifact。

## 兼具学术价值与生产价值的方向

### 1. 让 native operation 携带可独立检查的证书

第一个 artifact 可以是一套小型 delegated BPF operation certification interface。

每个 operation 携带 verifier-visible proof sequence、稳定 semantic identity、明确 effect summary、native implementation，以及由一个明显小于 optimizer/code generator 的 checker 验证的 certificate。certificate 可以是 restricted operation language 上的 proof object，也可以是 translation-validation evidence；重点是 checker 可以独立审计。

最强 baseline 不只是“不开优化”。应该同时比较 stock BPF/JIT、只靠人工 review 的 native module，以及把整个 generator 当 trusted component 的 native operation。评测 accepted optimization coverage、checker latency、checker/TCB size、speedup，以及每增加一个 operation/backend 带来的 trusted code。然后主动注入错误：算错寄存器、破坏 callee-saved state、越权 memory write、漏 side effect、坏 encoding、backend corner case。

学术问题是 extensible JIT 的 trust 能不能只随着一个小 checker 增长，而不是随着所有 optimizer/backend 增长。生产价值则是允许快速演化甚至第三方提供的 native optimization，而不必把整套 toolchain 都放进 kernel trust boundary。

如果 checker 最后复杂到接近 optimizer，或者大多数有价值的 operation 都无法用紧凑语义表示，这个方向就不值得。

### 2. 给无法完整证明的委托加 effect envelope

有些 native operation 可能太复杂，完整证明成本很高。第二个方向仍可以先限制 failure radius。

从 verifier-visible semantics 生成 effect envelope，限定它能输出哪些 register、访问哪些 memory region、调用哪类 helper/kfunc、怎样改变 reference lifetime，以及如何返回控制流。static validation 先拒绝 decoded effect 明显越界的 machine code；只有 value-dependent effect 无法静态决定时，runtime 才插入窄 guard。高风险 operation 在部署初期或 kernel/JIT update 后，还可以对 portable BPF 做小比例 differential canary。

这里不是解决跨架构 fallback。目标机器已经能运行这个 native operation。问题是 implementation 有 bug 时能不能逃出被委托权限。一旦 guard 违规或 differential mismatch，runtime 应禁用这一代 implementation，并记录 exact artifact 与失败原因。

评测要故意制造坏 native operation，比较 full proof、static effect checking、bounded runtime containment、shadow execution 和 review-only trust。主要指标包括 semantic escape、检测延迟、guard overhead、false positive，以及多少 operation 在没有 full proof 时仍能被有效限制。

如果 containment 成本比优化省下的时间还高，或者重要错误大多发生在 declared envelope 内，这个方向就失去价值。

### 3. 做一个专门攻击 TCB 的 eBPF specialization benchmark

现在的 optimization evaluation 自然会测 verifier acceptance、code size 和 runtime。trust-boundary benchmark 应该反过来，主动让 optimizer 出错。

从 verifier 已接受的 BPF program 与正常 certified native operation 开始，系统化生成 mutation：错误 arithmetic flag、过期架构假设、register clobber、越权 load/store、漏 reference release、多余 helper call、异常 case 错误、被篡改 machine-code artifact 与恶意 certificate。然后分别在 stock BPF/JIT、kfunc-style typed boundary、review-only delegation、proof-linked delegation、proof-linked + effect containment 下运行。

第一指标应该是 **escaped semantic violation**，而不是 crash rate。只要 native execution 产生 verifier-approved semantic program 不可能产生的可观测状态，就说明边界失守。第二指标再看每增加一个 operation/backend 带来的 trusted-code growth、certification latency、optimization coverage、runtime overhead、拒绝故障后留下的解释证据，以及 artifact identity 是否足够复现问题。

这样的 benchmark 能让系统论文不只写“safe by construction”，而是展示 construction 实际挡住哪些 fault、trust 还在哪里累积。生产环境也可以把 mutation corpus 用作新 JIT backend 或 operation library 的 regression test。

如果独立 implementation 几乎从不暴露普通 kernel testing 发现不了的错误，这项 benchmark 同样会给出有价值的反结论：继续依赖简单 review-and-test，而不是增加新的 certification layer。

## 哪些结果会改变这个判断？

如果有价值的 specialization 基本都能留在 verifier-visible BPF 世界里，单独设计 native-operation trust contract 的必要性会明显下降。BPF-level superoptimizer 说明这条简单路径为什么有吸引力：优化后的 bytecode 仍可走已有 verifier 与 stock JIT。如果 native operation 对这些 rewrite 的额外收益很小，就没必要移动 trust boundary。

如果 native delegation 最终只包含少量、长期稳定、完全 in-tree 的 mechanism，结论也会变弱。Linux 本来就依赖 code review 信任 kfunc implementation 与 JIT backend。如果扩展接口始终不会成为第三方 operation surface，现有 kernel trust model 可能已经足够。

最后，真实 mutation study 也可能证明 small checker 几乎没增加保护。如果绝大多数 realistic implementation fault 已经会被 compiler validation、architecture test、verifier-side contract、普通 kernel review 与 differential testing 捕获，那么 certificate complexity 就不划算。

但 extensible optimizer 改变了规模假设。kfunc 说明 verifier 可以在 trusted call boundary 上强制精确义务；Jitterbug 说明 deployed BPF JIT 里确实存在能被精确等价规格发现的语义 bug；Kops 则说明 native operation 可以继续保留 verifier-visible semantic witness，而更大的 native replacement 会用更大 TCB 换速度。**下一步值得做的抽象，是把这笔 trust 成本变得显式、可检查：只有当独立证据把 exact native artifact 与 verifier 已批准语义绑定起来时，才把对应权限委托给它。**

## 参考资料

- Linux kernel documentation. [eBPF verifier](https://www.kernel.org/doc/html/latest/bpf/verifier.html)，访问于 2026-09-08。
- Linux kernel documentation. [BPF Kernel Functions (kfuncs)](https://www.kernel.org/doc/html/latest/bpf/kfuncs.html)，访问于 2026-09-08。
- Linux kernel documentation. [BPF signing](https://www.kernel.org/doc/html/latest/bpf/signing.html)，访问于 2026-09-08。
- Luke Nelson, Jacob Van Geffen, Emina Torlak, Xi Wang. [Specification and verification in the field: Applying formal methods to BPF just-in-time compilers in the Linux kernel](https://www.usenix.org/conference/osdi20/presentation/nelson), OSDI 2020.
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
