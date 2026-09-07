---
date: 2026-09-07
title: "eBPF 加入 Native 快路径后，可信计算基最小能缩到多小？"
description: "eBPF proof sequence 能通过 verifier，但 CPU 最终执行的是 native emit。本文讨论如何用加载时验证、受限 emitter 与故障注入 benchmark 缩小可信计算基。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - Compilers
  - Program Verification
  - Kernel Security
research_question: "eBPF 怎样利用特定架构的 native fast path，同时又不把整个 optimizer 和 machine-code emitter 都变成 kernel 的 trusted computing base？"
source_cutoff: 2026-09-07
status: daily-report
---

# eBPF 加入 Native 快路径后，可信计算基最小能缩到多小？

假设一个 eBPF optimizer 识别出了 rotate、bit-select、checksum idiom，或者另一种 stock JIT 降得不够好的操作。它先给出一段普通 BPF proof sequence，verifier 接受这段程序，然后 architecture-specific emitter 再把它替换成更短、更快的 native instruction sequence。

这里有一个很容易被忽略的断层：verifier 可以完全正确，但 emitter 只要写错一个寄存器、用错一个 stack offset，或者生成了一个和 proof sequence 不一致的 control-flow target，程序仍然会错。CPU 执行的是 native bytes，不是 verifier 抽象解释过的 BPF 指令。

所以 native specialization 本质上是在转移 trust。上一篇关于[架构特定 eBPF 优化可移植性](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)的报告讨论的是：某个 native implementation 在这台机器上是否 eligible，以及不支持时怎样回退到同一份 portable semantic witness。今天的问题更往下一层：**当 eligible native implementation 已经存在之后，系统到底还必须额外信任多少代码，才能声称这些机器码仍然实现 verifier 已经批准的 operation？**

答案大概不可能是零。如果 kernel 接受任意 native bytes 而不做检查，那么 code generator 或 proof chain 总要有一部分属于可信边界。但我们没有必要把完整 userspace optimizer、整个 compiler backend、所有 optimization pass 和每个 architecture emitter 全部放进 TCB。更实际的目标是做一个 **bounded trust envelope**：optimization discovery 留在 TCB 外，把 native-operation interface 限制住，并要求最终 native bytes 在替换 portable BPF 之前先通过一个更小、可以独立检查的 validation boundary。

<!-- more -->

## Verifier 证明的是 BPF semantics，不是任意 native bytes

Linux eBPF 把 admission 和 execution 分开。verifier 检查 BPF program，architecture JIT 再生成 native code。这样 verifier 的安全推理不需要绑定某一种 CPU ISA，但也自然产生了一个 compiler-correctness boundary：只有 JIT 保持了被验证过的 BPF semantics，最终 native execution 才是正确的。

现实里这个边界确实会出错。OSDI 2020 的 Jitterbug 为 Linux BPF JIT 建立了精确的 correctness specification，并在五个已经部署的 JIT 里找到并修复了 **16 个此前未知的 bug**。真正重要的不是“以前的 JIT 很差”，而是 verifier acceptance 和 JIT correctness 本来就是两个不同的性质。BPF 程序本身完全安全，backend 仍然可能把它翻译错。

2026 年的 kernel 开发也持续暴露同一种结构问题。4 月，Linux 把 constant blinding 从各个 architecture-specific JIT 搬到了 generic verifier-side code。原因不只是重复代码难看：JIT 本地修改自己的 instruction copy 后，可能让这份 instruction stream 和 global verifier auxiliary metadata 不再同步。新的做法是在更统一的位置做 rewrite，并同时调整相应 verifier state。

这给了一个很明确的设计信号。如果某个 transformation 会改 instruction stream，又依赖 verifier 已经推导出来的事实，那么把它分别复制进每个 backend，就等于让更多地方各自维护隐藏 invariant。把共同的 semantic transformation 收回到一个统一边界，可以同时减少 inconsistency risk 和 per-architecture trusted logic。

## Native operation 把 trust transfer 直接暴露出来了

[Kops](https://arxiv.org/abs/2606.24213) 对这个边界的表达非常清楚。一个 Kops operation 有两种形式：一段由现有 verifier 检查的普通 eBPF proof sequence，以及一段提供 architecture-specific machine instructions 的 native emit。论文里的 EInsn 用 Lean 4 proof 证明 native implementation 和 proof sequence 计算同一个结果。它在 microbenchmark 上最高提升 24%，在 production application 上最高提升 12%。

这个设计最有吸引力的地方，是 userspace optimizer 不会因为“发现了优化机会”就自动进入可信边界。proof sequence 仍然是普通 BPF，而 native implementation 可以保持很小。

但 production 里还多一层。如果 kernel 因为某个 module 注册了 native emit 就直接使用它，那么真正生成机器码的 emitter 仍然在 trusted path 里。离线 Lean proof 能很好地说明被建模实现的等价性，但 operator 还需要知道：今天真正选中的 bytes，是否确实由那份被证明过的实现产生；architecture feature、clobber、calling convention、stack layout 和 kernel/JIT assumption 是否还是 proof 所针对的条件。

whole-program native replacement 会让这个 trade-off 更明显。Kops 也能把完整 BPF program 换成 native code，并报告最高 2.358x，但论文也明确指出这会带来更大的 TCB。也就是说，性能越往上走，semantic boundary 可能恰好越难审计。

## 最新 KASAN 工作说明：安全插桩自己也可能变成 codegen hazard

本周刚进入 `bpf-next` 的一个例子特别适合说明这个问题。2026 年 9 月的 v9 patch series 给 JITed BPF program 增加 KASAN check：识别有 memory access 的 BPF instruction，然后让 x86 JIT 在相应位置插入 KASAN instrumentation。

它的目标显然是增强诊断和内存安全，但 v9 cover letter 同时记录了 instrumentation 自身的一个 bug：一个本来不该插桩的 stack access 不仅被 instrument 了，而且还可能指向**错误的 stack offset**。v9 因此又增加了一层 guard：除了 verifier 提供的 stack-access metadata，还在 `emit_kasan_check` 里对 BPF frame pointer 和 parameter register 做系统性跳过。

这并不是反对 JIT-side KASAN。恰恰相反，它给了一个非常干净的 trust counterexample。即使一段 codegen 的目的就是提高 memory-safety diagnosis，它仍然必须正确保持 register state、stack identity、instruction offset、被 patch 后的 verifier metadata，以及 architecture JIT 的 calling convention。backend transformation 携带的 semantic knowledge 越多，一个局部看起来合理的 emit 就越有可能破坏全局 invariant。

所以对 native eBPF fast path 来说，问题不能只停在“这个 operation 是否曾经被证明过”。更值得问的是：**每次真正选择并 emit 这份 native implementation 时，最少有哪些机制必须正确？**

## 现有研究还缺什么

### Offline equivalence 还没有绑定到真正加载的 bytes

一个 proof 可以证明某个 implementation function 在模型里与 BPF proof sequence 等价。production loading 还多了 compiler version、target feature、relocation、kernel version、JIT state、module version，以及最终真正生成的 byte sequence。

如果系统不能把 proof result 绑定到实际执行的 operation identity 和 native bytes，这份 proof 仍然很有价值，但作为 deployment artifact 还不完整。reproducible build 可以减少供应链差异，却不会自动告诉 kernel：这段机器码允许修改哪些寄存器、有哪些 memory effect、可以跳到哪里，以及 admission 前应该检查什么。

### 任意 native emitter 暴露了太大的 machine-state surface

一个 unrestricted emitter 可以写 register、stack slot、memory、branch target 和 helper-call state。源代码看起来只是一个很短的 function，它的 semantic effect surface 仍然可能非常大。

因此不能简单把“一个 native emit function”当成 TCB 大小。真正有用的 boundary 至少要写清楚：哪些 register 可以变化、哪些 memory 可以访问、control flow 能不能离开 operation、是否允许 helper/kfunc call、哪些 stack range 合法，以及需要哪些 CPU feature 或 calling convention assumption。

### Optimization 论文很少把 trusted-code growth 和 fault escape 放在同一张图里

speedup 很容易画图，TCB 往往只做定性描述。这让 stock JIT optimization、Kops-style local native operation、verified compiler backend 和 whole-program native replacement 很难公平比较。

真正缺的实验应该带攻击性：故意 mutate emitter、metadata 或 proof linkage，然后测 admission boundary 能不能在执行前拒绝错误 implementation。否则所谓 “small TCB” 很容易退化成“代码行数比较少”，而不是一个被评估过的 correctness/security property。

## 有学术和生产价值的方向

### 1. 做 load-time proof-carrying native operation capsule

每个 native operation 可以带一个 capsule，把五类 identity 绑在一起：

```text
semantic operation ID
        + portable BPF proof sequence hash
        + target/feature predicate
        + declared native effects
        + exact emitted-code hash + validation certificate
```

userspace optimizer 可以完全不可信。它可以负责搜索 candidate、决定什么时候值得尝试 specialization，但 kernel 只有在一个更小的 checker 确认 capsule 属于被批准的 semantic operation，而且声明的 machine effect 与 operation contract 一致时，才接受 native implementation。

具体实现可以有几种。formally proved emitter 可以生成 compact certificate，由更小的 checker 验证；translation validator 可以在 load time 比较一段 bounded native sequence 与 BPF operation semantics；对于很短的 hardware idiom，kernel 甚至可以只接受预定义 instruction template 加 relocation，而不是任意 byte array。

研究难点是 checker 必须明显小于它从 TCB 中替代掉的 compiler。如果 validator 最后又长成一个完整 optimizing compiler，只是把 trust problem 换了个名字。

prototype 可以先从 Kops 同类 operation 开始，例如 rotate、conditional select、byte manipulation 等 register-local idiom。它们的 semantic state 足够小，适合做精确 validation，同时又确实能使用 architecture-specific instruction。

### 2. 用 effect-typed native emitter 代替 unrestricted assembler

第二个方向是直接限制 emitter 可以表达什么。

可以设计一个很小的 native-operation IR 或 typed macro-assembler，每个 primitive 都携带 effect，例如：

- 读取 `r1`, `r2`，只写 `r0`；
- 可以 clobber flags，但不能改声明之外的 BPF-visible register；
- 不访问 memory，或只访问声明过的 stack range；
- 不允许 indirect branch；
- 不调用 helper，或者只能按指定 ABI 调用某个 named helper；
- 明确要求 `x86_64 + BMI2`、`arm64 + LSE` 或其他 target feature。

最终 architecture backend 的任务就窄很多：把已经带类型和 effect 的 primitive encode 成 bytes。loader 则可以机械检查 control-flow shape、stack access、clobber 和 feature predicate。

这种机制不会自动证明所有 arithmetic identity，因此它应该和 semantic-equivalence proof 配合，而不是替代 proof。它主要约束那些位于 algebraic proof 之下的错误：wrong register、undeclared memory access、wrong stack offset、invalid branch target、ABI mismatch。

最近 KASAN JIT 的修复让这个方向非常具体。如果系统真的声明“这段 instrumentation 不得访问 BPF stack”，typed emitter 应该把它变成 machine-checkable effect rule，而不是让这条约束分散在 verifier metadata 和 architecture code 里的约定。

### 3. 做一个带 emitter fault injection 的 TCB-performance frontier benchmark

evaluation 应该把 trust cost 变成第一等指标，而不是 appendix。

选一组 XDP、tracing 和 compute-oriented BPF workload，至少比较四种配置：

1. stock Linux JIT；
2. 使用 manually trusted emitter 的 local native operation；
3. 同样的 operation，但增加 load-time validation 和 effect typing；
4. whole-program native replacement，或者明显更 expressive 的 native backend。

至少在 x86-64 和 ARM64 上跑，并覆盖多个 kernel/JIT version。每个 configuration 除了 throughput/latency，还测 specialization coverage、load-time validation cost、trusted implementation size，以及 architecture-specific trusted component 数量。

然后主动注入错误：改 destination register、stack displacement、immediate、branch target、CPU-feature predicate、clobber declaration、operation identity 或 proof/code hash。第一优先级 correctness metric 是 **mutant escape rate**：有多少 semantic 上错误的 implementation 能穿过 admission boundary 并真正执行。同时也记录正确 native implementation 被误拒绝的比例。

最终得到的是一条 TCB-performance frontier。一种设计可能只快 10%，但只增加一个很小的 checker，而且几乎所有注入错误都会被挡住；另一种设计可能快 30%，却要求信任一个可以任意读写 memory 和 control flow 的大 backend。这个结果比单独一张 speedup graph 更接近真实系统决策。

## 实际需要的是分层 trust，而不是一个什么都验证的万能 verifier

目标不应该是强迫现有 Linux eBPF verifier 直接理解所有 x86 或 ARM instruction。那会把 portable BPF admission 和 architecture semantics 缠在一起，让 verifier 本身越来越难维护。

更干净的层次是：

```text
portable BPF program
      |
      v
Linux verifier                 <- 现有 portable safety boundary
      |
      v
semantic native-operation ID
      |
      +--> 没有 eligible implementation -> 执行 portable BPF form
      |
      v
small native validator/checker <- bounded extra trust boundary
      |
      v
exact native bytes + provenance
```

architecture-specific optimizer、profile collector、search algorithm 或 superoptimizer 都可以留在这条 trusted path 外面。它们如果提出一份垃圾 candidate，最坏结果应该是 optimization miss，而不是 kernel memory corruption 或静默 semantic change。

这也顺手建立了更好的 debugging foundation。operator 问“到底跑了哪段代码”时，runtime 可以给出 portable operation、selected native implementation、target predicate、checker result 和 code hash。debugging/provenance 不是今天的主问题，但 bounded trust interface 会给下一步工作一个稳定、可记录的 identity。

## 哪些结果会改变这个判断？

有几类结果会削弱单独设计 bounded native-operation trust layer 的必要性。

第一，如果 production measurement 发现绝大多数有价值的 eBPF speedup 都能通过普通 BPF-to-BPF rewrite 获得，并继续走 stock verifier 和 JIT，那么 native emit 增加的 trust surface 可能根本不值得。前一篇[运行时 profile specialization](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/)已经讨论了这个优先级：portable form 能表达优化时，先不要越过 native boundary。

第二，未来如果出现覆盖主要架构的 broadly verified JIT，per-operation validation 可能没有必要。Jitterbug 已经证明 production JIT component 可以被形式化验证。如果 Linux 后续相关 backend 能获得 machine-checked code generation，而 native-operation interface 又可以直接复用这套 verified backend，不再引入任意 emit logic，那么额外 capsule checker 可能是重复建设。

第三，提议的 validator 也可能在 adversarial benchmark 里失败。如果 realistic wrong-register、stack-offset、control-flow 或 proof-linkage mutation 经常逃过一个 compact checker，那说明 bounded-operation abstraction 还不够强。系统应该引入更强的 proof boundary，或者干脆回退到 portable BPF，而不是继续声称安全。

最后，如果 load-time proof checking 或 translation validation 的开销足够大，直接吃掉了短 native operation 的性能收益，那么正确 granularity 可能不是“一条 operation 一个 certificate”，而是更大的 verified region。

目前 evidence 更支持中间路线。Linux 的历史说明，JIT correctness 不能从 verifier correctness 自动推出；Kops 说明有价值的 native operation 可以远小于整个 compiler backend；最近 constant blinding 和 KASAN 的工作又说明，backend-local transformation 即使是为 security hardening 服务，也会因为 verifier/JIT invariant 很细而出错。**因此真正有意思的问题不是怎样去信任一个更快的 compiler，而是怎样设计 interface，让系统一开始就不需要信任这个 compiler 的绝大多数部分。**

## 参考资料

- Yusheng Zheng 等，[Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213)，2026。
- Luke Nelson 等，[Specification and verification in the field: Applying formal methods to BPF just-in-time compilers in the Linux kernel](https://www.usenix.org/conference/osdi20/presentation/nelson)，OSDI 2020。
- Xu Kuohai，[Move constants blinding out of arch-specific JITs](https://lists.openwall.net/linux-kernel/2026/04/15/79)，Linux BPF patch discussion，2026。
- Linux BPF maintainers，[Applied ENDBR/BTI and generic constant-blinding series](https://lists.openwall.net/linux-kernel/2026/04/16/938)，2026。
- Alexis Lothoré，[KASAN checks in JITed BPF programs, v9](https://lkml.iu.edu/2609.0/08000.html)，2026。
- Linux BPF maintainers，[Applied KASAN support for JITed BPF programs](https://lkml.iu.edu/2609.0/11290.html)，2026。
