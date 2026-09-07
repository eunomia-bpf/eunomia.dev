---
date: 2026-09-07
title: "eBPF 把操作交给原生代码后，怎样控制可信边界？"
description: "eBPF 原生操作可能绕过 verifier 可见的字节码，并扩大可信计算基。本文分析怎样用语义证明、独立检查和故障测试控制这种委托。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - 程序验证
  - 系统安全
  - 编译器
research_question: "eBPF 怎样把已经验证的语义操作委托给原生代码，同时避免把每个优化器、后端和原生实现都纳入可信计算基？"
source_cutoff: 2026-09-07
status: daily-report
---

# eBPF 把操作交给原生代码后，怎样控制可信边界？

假设一个 eBPF 优化器发现，原本需要多条 BPF 指令完成的操作，在某种 CPU 上可以用一条原生指令实现。原来的 BPF 序列已经通过 verifier，目标机器也支持这个快路径，loader 还正确选中了对应实现。此时仍然少了一个问题：**谁来证明这段原生代码只做了那份 BPF 程序被允许做的事情？**

这个问题不能靠 verifier 自动解决。verifier 分析的是 BPF 指令，以及它已知语义的函数调用；优化器之后插入的任意机器码并不在这套分析里。一旦 specialization 允许用原生代码替换 verifier 看过的逻辑，原生实现里的一个错误就可能变成内核错误，即使最初的 BPF 程序完全安全。

本文的判断是：**原生委托应该同时携带 verifier 可见的语义依据，以及可以独立检查的实现证据。这样，新增一个优化不必自动把它的优化器和代码生成器一起放进可信计算基。**

<!-- more -->

上一篇关于[特定架构 eBPF 优化可移植性](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)的报告讨论的是：某个原生实现能不能在这台机器上使用，如果不能应该怎样安全退回 portable BPF。那是 capability 和 portability 问题。本文假设目标实现已经正确选出来，只追问它获得与原始 BPF 相同执行权限的理由是否可信。

更早一篇关于[运行时画像驱动 eBPF specialization](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/)的报告讨论 workload assumption 会不会过期。本文的错误不需要等 workload 改变才出现。一段实现错误的 native emit 第一次执行就可能偏离原始语义。

## Verifier 可以约束调用边界，却不会自动证明被调用函数

Linux 的 BPF kernel function，也就是 kfunc，是一个很好的现成例子。kfunc 把内核函数暴露给 BPF 程序。当前 Linux 文档明确体现出两个层次。

第一层是 verifier 可以检查相当丰富的调用契约。指针参数默认需要可信，BTF 类型限制可以传入什么对象，而 `KF_ACQUIRE`、`KF_RELEASE`、`KF_RET_NULL`、`KF_SLEEPABLE`、`KF_RCU` 和 `KF_DESTRUCTIVE` 等标记告诉 verifier 资源生命周期、空指针、执行上下文和危险 effect 应该怎样处理。例如，一个 `KF_ACQUIRE` kfunc 返回引用以后，verifier 会继续检查这份引用最终是否被释放，或者被转移进支持的 map 状态。

第二层是调用边界后面的实现仍然是普通内核代码。Linux 文档说明，已有内核函数可以直接注册成 kfunc，但维护者仍然需要审查 BPF 在什么上下文调用它，以及这样做是否安全。verifier 能保证参数有效，并追踪已经声明的属性；这些声明本身并不会证明函数体实现正确。

这种划分本身很合理。否则 verifier 为了验证一个 BPF 程序，就得理解整个 Linux 内核。但是，当 userspace optimizer 或第三方模块可以不断加入新的机器相关操作时，这个划分也把 trust 问题暴露得很清楚：边界契约只有在委托实现真的遵守契约时才有效。

通用的 [eBPF verifier 文档](https://www.kernel.org/doc/html/latest/bpf/verifier.html)也体现同样的关系。verifier 对 BPF 指令做符号执行，对函数调用检查参数约束，并在调用后按照函数原型更新寄存器状态。这是一条 interface-level proof boundary，而不是对 callee 机器码做语义等价证明。

## Kops 把新增的可信代码明确暴露出来

[Kops](https://arxiv.org/abs/2606.24213) 很适合分析这个问题，因为它直接讨论了这笔 trade-off。每个 Kops operation 都有两种表示：一份由普通 eBPF 指令组成、可以交给现有 verifier 检查的 proof sequence；以及一份架构相关 JIT 可以使用的 native emit。普通序列仍然处于 verifier 的语义世界里，native emit 则是每个 operation 新增的可信实现。

对于文中实现的硬件 idiom，Kops 使用 Lean 4 证明 native emit 与 BPF proof sequence 等价。论文报告在 microbenchmark 上最高 24% 的加速，在生产应用上最高 12%。它还演示了 whole-program native replacement，在评测中达到 2.358x，但论文也明确指出这会换来更大的 trusted computing base（TCB）。

这使工程选择变得更具体。原生委托不是只有“快或者不快”一个维度。它留下多少 verifier 可见语义、又把多少机器码放进必须信任的范围，直接决定了安全成本。

Linux 对 BPF JIT 的控制也从另一个角度说明原生代码是安全敏感对象。内核提供 JIT hardening 来降低 JIT spraying 风险，也提供面向特权用户的 JIT 代码观察机制。这些措施不证明语义等价，但它们说明最终机器码并不是一个可以忽略的实现细节。

## 真正缺少的是语义权限与实现权限之间的契约

这里最好把两个经常混在一起的问题分开。

**语义权限**定义一个 operation 被允许做什么。对原生 BPF operation 来说，它可以由一份普通 BPF proof sequence 加上一份明确的 effect summary 表示：哪些寄存器是输入和输出，可以读写哪些 context 或 map memory，是否允许调用 helper/kfunc，是否可能 sleep、分配资源、取得引用或者产生外部可见 effect。

**实现权限**则回答：为什么这一段具体机器码可以代替上面的语义操作。它应该把一个确定的实现 hash 与语义依据绑定，再附上可以由独立 checker 验证的证据，而不是只相信生成它的 optimizer。

一个最小的表示可能类似：

```text
semantic_op = rotate64_v1
proof_seq_hash = sha256(portable_bpf_sequence)
effects = {read: r1,r2; write: r0; memory: none; calls: none}

native_impl_hash = sha256(machine_code)
target = x86_64
certificate = proof-or-translation-validation-result
checker_version = v3
```

具体字段并不是本文要提前冻结的 ABI。关键在于职责分离：userspace optimizer 可以大胆搜索优化，backend 可以生成机器码，而一个远小于 optimizer 的 checker 决定产物是否能获得执行权限。如果 optimizer 出错，应该是 checker 拒绝它，而不是因此把整个 optimizer 变成内核安全的一部分。

这与上一篇的 capability manifest 不同。capability negotiation 回答“这个实现能不能在这台机器上运行”；trust contract 回答“为什么这份实现可以获得与 verifier 已批准语义相同的权限”。

## 现有研究还缺什么

### kfunc 式契约能约束边界，却不能证明实现等价

Linux kfunc 的 metadata 已经可以表达 pointer trust、ownership、RCU 状态、sleepability 和 destructive behavior。verifier 因此可以拒绝许多不安全调用。但边界后面仍然依赖内核代码审查和实现本身正确。

对于少量、稳定、in-tree 的 kernel function，这可能就是合理的工程取舍。但如果将来 userspace optimizer 或第三方模块可以加入大量 machine-specific operation，单靠 review 的可信范围会随着 operation 数量和 backend 数量一起扩大。

这里缺的不是更多类型标记，而是一项故障实验：保持 kfunc-like signature 和声明 effect 完全不变，只故意修改 native body，看看 compact semantic/effect contract 加独立 checker 能捕获多少 call-site typing 捕获不了的错误。

### proof-linked operation 仍然需要可部署的 checker boundary

Kops 已经说明 native emit 可以与 verifier 可见的 proof sequence 建立形式关系，Lean 证明也为文中评测的 operation 提供了强证据。生产扩展机制还必须决定：什么在 build time 检查，什么在 load time 检查，什么在进入 kernel 时检查，以及哪些东西最终仍然只能信任。

如果系统因为“proof 是某个 compiler pipeline 生成的”就直接信任 proof，那么可信范围只是搬了位置。相反，如果一个很小的 proof checker 或 translation validator 能独立拒绝错误实现，生成器就可以留在 TCB 外面。

真正需要测的是 operation library 和 architecture matrix 扩大时，trusted-code growth 与 escaped semantic bug 怎样变化，而不是再做一次单纯的 speedup benchmark。

### whole-program native replacement 会扩大一次错误的影响范围

把一个算术 idiom 替换成一条机器指令，与把完整 BPF 程序替换成 native code 都可以叫 native specialization，但它们的 failure radius 完全不同。小 operation 可能根本不碰内存；完整程序则可能读写 packet/context、调用 helper、更新 map，并经过很多控制流状态。

因此 trust model 不能只有一个“equivalent”标签，还需要 effect envelope。如果系统暂时无法完整证明实现，至少应该知道 native body 获准使用哪些类别的 effect，并在越界时 fail closed。

## 兼具学术价值与生产价值的方向

### 1. 让每个 native operation 携带可独立检查的证书

第一个 artifact 可以是一套小型 delegated BPF operation certification interface。

每个 operation 携带 verifier 可见的 proof sequence、稳定 semantic identity、明确 effect summary、native implementation，以及由一个明显小于 optimizer/code generator 的 checker 验证的 certificate。证书可以是受限 operation language 上的 proof object，也可以是 translation-validation evidence；关键是 checker 本身能够独立审计。

最强 baseline 不应该只是“不开优化”。应该同时比较 stock BPF/JIT、只靠人工 review 的 native module，以及把整个 generator 当成 trusted component 的 native operation。评测 acceptance coverage、checker latency、代码体积、speedup，以及新增 operation/backend 带来的 trusted source lines 或 trusted component 数量。然后主动注入错误：算错寄存器结果、破坏 callee-saved state、越权内存写、漏掉 side effect、架构 corner case。

学术问题是：可扩展 JIT 的可信范围能否只随着一个小 checker 增长，而不是随着所有 optimizer/backend 实现增长。生产价值则是让快速演化、甚至第三方提供的 native optimization 不必整套进入 kernel trust boundary。

如果 certificate/checker 最后复杂到接近 optimizer 本身，或者绝大多数有价值的 operation 都无法用紧凑语义检查，这个方向就不值得。

### 2. 给无法完整证明的委托加 runtime effect envelope

有些 native operation 可能太复杂，完整证明成本很高。第二个方向仍然可以把故障范围收紧。

从 verifier 可见语义生成 effect envelope，限定它能输出哪些寄存器、访问哪些内存区域、调用哪类 helper/kfunc、怎样改变 reference lifetime，以及如何返回控制流。JIT 或 runtime 只在 native code 有机会越过这些边界的位置插入便宜 guard。高风险 operation 在部署初期或者 kernel/JIT 更新后，还可以用 portable BPF 版本做小比例 differential canary。

这里不是在解决跨架构 fallback。目标机器已经有能力执行这个 native operation。机制要解决的是：实现有 bug 时，它能不能逃出自己被委托的权限。一旦 guard 违规或 differential result 不一致，runtime 就禁用这一代 implementation，并保留明确的失败记录。

评测可以故意制造坏的 native operation，比较 full proof、effect-only containment、shadow execution 和 review-only trust。主要指标包括 semantic escape 数量、检测延迟、guard overhead、false positive，以及多少 operation 能在没有 full proof 的情况下被有效限制。

生产用户是频繁升级 kernel 和 optimizer、同时允许 extensible re-JIT 的 operator。如果 guard 成本比优化本身省下的时间还高，或者大多数真正语义错误都发生在已声明 envelope 内，这个方向就失去价值。

### 3. 做一个专门攻击 TCB 的 eBPF specialization benchmark

现在的 optimization evaluation 自然会测 verifier acceptance、code size 和 runtime。trust-boundary benchmark 应该反过来，主动让 optimizer 出错。

从 verifier 已接受的 BPF 程序与正常 native operation 开始，系统化生成受控 mutation：错误 arithmetic flag、过期架构假设、register clobber、越权 load/store、漏掉 reference release、多余 helper call、异常 case 错误，以及恶意 certificate。然后分别在 stock BPF/JIT、kfunc-style typed boundary、review-only native delegation、proof-linked delegation，以及 proof-linked + effect containment 下运行。

第一指标应该是 **escaped semantic violation**，而不是 crash rate。只要 native 执行产生了 verifier-approved semantic program 不可能产生的可观测状态，就算系统没有守住边界。第二指标再看每增加一个 operation/backend 带来的 trusted-code growth、certification latency、optimization coverage、runtime overhead，以及拒绝后能留下多少 debugging evidence。

这样的 benchmark 能让系统论文不只写“safe by construction”，而是展示 construction 实际挡住哪些错误、trust 还在哪里累积。生产环境也可以把同一套 mutation corpus 用作新 JIT backend 或 operation library 的回归测试。

如果独立实现几乎从不暴露普通 kernel testing 发现不了的错误，这项 benchmark 也会给出有价值的反结论：继续依赖更简单的 review-and-test 流程，而不是增加新的 certification layer。

## 哪些结果会改变这个判断？

如果有价值的 specialization 基本都能留在 verifier 可见的 BPF 世界里，那么单独设计 native-operation trust contract 的必要性会大幅下降。EPSO 一类 BPF-level optimizer 正说明了这种简单路径为什么有吸引力：优化后的 bytecode 仍然能走已有 verifier 和 stock JIT。如果 native operation 对这些 rewrite 的额外收益很小，就没有必要移动 trust boundary。

如果 native delegation 最终只包含很少量、长期稳定、完全 in-tree 的 mechanism，这个结论也会变弱。Linux 本来就依赖 code review 来信任 kfunc implementation 和 JIT backend。如果扩展接口始终不会变成第三方 operation surface，现有 kernel trust model 可能已经足够。

最后，一项真实 mutation study 也可能证明 small checker 没有带来多少额外保护。如果绝大多数 realistic implementation fault 已经会被 compiler validation、architecture tests、verifier-side contract 和普通内核 review 捕获，那么 certificate complexity 就不划算。

但 extensible optimizer 改变了规模假设。kfunc 说明 verifier 可以在 trusted call boundary 上强制很多精确义务；Kops 则说明 native operation 可以继续保留 verifier-visible semantic witness，并且更大的 native replacement 会用更大的 TCB 换速度。**下一步值得做的抽象，是把这笔 trust 成本变得显式、可检查：只有当独立证据把 native implementation 与 verifier 已批准语义绑定起来时，才把对应权限委托给它。**

## 参考资料

- Linux kernel documentation. [eBPF verifier](https://www.kernel.org/doc/html/latest/bpf/verifier.html)，访问于 2026-09-07。
- Linux kernel documentation. [BPF Kernel Functions (kfuncs)](https://www.kernel.org/doc/html/latest/bpf/kfuncs.html)，访问于 2026-09-07。
- Linux kernel documentation. [Documentation for /proc/sys/net/](https://kernel.org/doc/html/latest/admin-guide/sysctl/net.html)，访问于 2026-09-07。
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
- Qian Zhu et al. [EPSO: A Caching-Based Efficient Superoptimizer for BPF Bytecode](https://arxiv.org/abs/2511.15589), 2025.
