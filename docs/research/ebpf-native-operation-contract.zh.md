---
date: 2026-09-06
title: "eBPF 加原生硬件操作时，能不造第二个 Verifier 吗？"
description: "Native operation 能补回 eBPF 的硬件性能，但每个 backend 都会增加新的信任边界。本文提出可审计、可回退的 operation contract。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - Program Verification
  - Compilers
  - Kernel Security
research_question: "eBPF 怎样暴露架构相关的 native operation，同时让 verifier 可见的语义继续作为唯一权威，并把每个 JIT backend 新增的信任范围限制成可验证、可测试、可撤销的小边界？"
source_cutoff: 2026-09-06
status: daily-report
---

# eBPF 加原生硬件操作时，能不造第二个 Verifier 吗？

64 位 rotate 是一个很适合拿来测试 eBPF 编译链的问题。现代 CPU 一条 native instruction 就能完成，但 portable BPF bytecode 可能要展开成一串指令，而刻意保持简单的 kernel JIT 又可能把这种低效一路保留到最后的 machine code。

最直接的优化是让 JIT 识别这段序列，直接发出对应的硬件指令。真正麻烦的问题是：**谁来证明这条新的 machine instruction，仍然等价于 verifier 当初批准的程序？**

当 native operation 不再只是算术，这个问题会更难。某个架构的 lowering 可能依赖 CPU feature、control-flow hardening、memory ordering、verifier metadata、stack layout，或者 portable BPF sequence 里根本看不到的其他条件。如果每个 backend 都可以自己解释这些义务，系统实际上就在 verifier 后面又建立了一套新的语义权威。

本文主张一个更窄的边界：**native operation 应该是 verifier 可见的 portable contract 的一个实现，而不是新的 BPF 语义原语。** Contract 应明确 portable proof sequence、可观察 effect、architecture precondition、native implementation identity，以及证明这个实现正确的 evidence。实现不支持或失效时，应退回 portable sequence，而不是放松 verifier 边界。

<!-- more -->

这和昨天的 [runtime profile specialization](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/) 不同。昨天讨论的是 workload assumption 随时间变化时，重写后的 BPF program 什么时候还算同一个程序。今天可以假设 source program 和 workload 都不变，问题只集中在：一个 architecture-specific native implementation，是否真的是已经定义好的 BPF operation 的可信实现。

## Portable BPF ISA 是最合适的语义锚点

IETF 的 [BPF ISA 标准 RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) 给 BPF 提供了 portable instruction-level contract。算术位宽、signedness、atomic、jump、call 和 memory behavior 都有独立于某一个 x86-64 或 arm64 lowering 的语义。

这个分层很有价值，因为 Linux verifier 在 architecture JIT 生成 machine code 之前，就可以对 BPF-level state 做推理。Verifier 可以证明 register、pointer、control flow、helper use 与 bounded execution 等性质，不需要同时证明每个 backend 生成的每一段 native instruction sequence。

如果新的优化绕过这种关系，这个边界就会变弱。一个 verifier 无法理解的 backend-only pseudo operation，要么迫使 verifier 信任它没有检查过的 backend 逻辑，要么要求系统给新 operation 再做第二个 verifier。两种方法都会扩大 semantic TCB。

更合理的 extension point 应继续让 verifier 的语言保持权威。Native operation 可以有优化后的 machine implementation，但必须同时存在一份普通 BPF 表示，明确说明它到底是什么意思。

## Kops 说明 proof sequence + native emit 这个拆分可以工作

[Kops](https://arxiv.org/abs/2606.24213) 实现了这个基本结构。每个 operation 有两种形式：一段由普通 BPF instruction 组成、交给现有 verifier 检查的 proof sequence，以及 architecture JIT 可以生成的 native emit。EInsn prototype 包含 rotate、conditional select 等七种 hardware idiom。论文报告 microbenchmark 最高 24% 的提升、production application 最高 12% 的提升，并用 Lean 4 proof 把每个 native emit 和它的 proof sequence 对应起来。

这里最重要的不是这七条具体指令，而是 extension 不需要教 verifier 一条新的 opaque semantic rule。Proof sequence 仍然是 verifier 可见的含义，native emit 只是额外加入信任边界的具体实现。

Linux Plumbers Conference 2026 的公开 contribution [“kops and rejit: Safely Optimizing eBPF for Hardware and Workloads”](https://lpc.events/event/20/contributions/2445/) 也明确把问题放在这个 trust boundary 上：module-supplied native emit 到底应该怎样进入 kernel 的信任模型。

但这个设计仍然留下一个重要的系统问题：一个 native emit 在被允许替代 proof sequence 之前，究竟需要证明、检查或者测试哪些性质？

如果只是 rotate，destination register 的结果相等已经接近完整答案。但未来 operation 一旦涉及 memory、control flow、architecture state 或 concurrency，单纯比较输出值就远远不够。

## 当前 kernel JIT 的变化已经说明 backend 的义务不只是翻译指令

近期 Linux BPF 的变化给出了很具体的例子，说明某些信息不能只藏在 architecture backend 里面。

2026 年的一组 patch 把 [constant blinding 从 architecture-specific JIT 移到 generic verifier code](https://lists.openwall.net/linux-kernel/2026/04/15/79)。问题不是 constant blinding 算错了，而是 JIT 私有的 instruction rewrite 可能让已经变换过的 instruction copy 和 verifier-global auxiliary data 不同步。修复方法是把 rewrite 放到 generic verifier path，让 instruction 与 metadata 一起更新。

同一组 patch 还把 [`bpf_verifier_env` 传给 JIT](https://lists.openwall.net/linux-kernel/2026/04/16/307)。一个直接用途就是 control-flow integrity。x86 开 CET/IBT、arm64 开 BTI 时，indirect jump target 需要 architecture-specific landing-pad instruction。Verifier 已经知道哪些 BPF instruction 是 indirect target，因此 kernel 把这个事实传给 JIT，而不是让每个 backend 各自重新推导。随后这组 patch 已经进入 BPF tree，并加入 x86 ENDBR 与 arm64 BTI 支持。

这对 native operation 是一个很好的先例。Architecture-specific machine code 并不只是 `BPF opcode -> instruction bytes` 的函数。正确 lowering 可能依赖 verifier facts 与 platform security mode。一个安全 extension interface 必须能够显式表达这些 dependency。

JIT hardening 又增加了一层义务。当前 Linux 文档暴露了 `bpf_jit_harden`，2026 年 kernel fix 也增强了 JIT spraying 与 JIT memory reuse 时 indirect-branch predictor 的保护。一个 native operation 完全可能在返回值上和 proof sequence 一样，却因为发出了错误的 control-transfer shape 或绕过 mitigation，而违反 code-generation hardening invariant。

## Concurrency 让“结果一样”变成更弱的测试

LPC 2026 即将讨论的 [blitmus](https://lpc.events/event/20/contributions/2432/) 指向另一个边界。BPF 已经包含 acquire/release、atomic、spinlock、ring buffer 等并发机制，而且这些语义需要跨 x86、arm64、RISC-V、PowerPC 和更多 JIT backend。该项目指出，BPF 目前仍缺少类似 LKMM 那样可执行的 formal memory model，也缺少端到端方法来检查程序请求的 ordering 是否真的经过 verifier 和各架构 JIT 保留下来。

这对 extensible native operation 很重要。两个实现可以在 single-thread test 里产生完全一样的 register value，但如果其中一个漏了 barrier，或者用了更弱 ordering 的 instruction，在 concurrency 下就会出现不同结果。所以有用的 conformance contract 需要的是 **effect model**，而不只是 input/output relation。

LPC 2026 的 [Proof-Carrying Verification for eBPF](https://lpc.events/event/20/contributions/2440/) 也体现了类似的设计压力。它把昂贵的 proof discovery 放在 userspace，让 kernel 内一个受限 checker 只验证显式 proof step。这个工作的目标是 verifier safety，而不是 JIT native operation，但结构值得借鉴：不可信 tooling 可以生成丰富 evidence，而 kernel 只保留小而确定的 checking surface。

## 现有研究还缺什么

### Native-operation proof 通常只定义了过窄的 observable contract

对纯 arithmetic idiom 来说，证明 destination register 和 portable sequence 一致很有说服力。但一个通用 extension mechanism 迟早会想支持 load、store、atomic、address-space property 或 control flow。到那时，observable contract 还包括 memory effect、ordering、fault、clobber、helper-visible state、control-flow target，甚至 speculation / hardening constraint。

如果每个 operation 都自己发明一套“等价”的定义，接口会越来越难审计，也很难组合。这里真正缺的是少量 BPF-specific effect class，让 native implementation 都在同一套 contract 下说明自己需要保持什么。

### Architecture 与 hardening precondition 还没有自然地成为 operation identity 的一部分

一个 native emit 可能只在特定 feature set 或 kernel configuration 下正确。例如某个 ISA extension、CET/IBT 或 BTI mode、JIT hardening、stack-layout constraint，或者 verifier 已知的 indirect-target identity。

如果这些条件只是 backend code 里的注释，fallback 和 incident response 会很脆弱。Operation 应该携带 machine-readable precondition，在 implementation 注册时检查，在真正选择执行时也检查。

### Cross-architecture conformance 仍然主要依赖每个 backend 自己维护

Kernel 有很多 BPF JIT backend。同一个 operation 今天可能只有 x86-64 和 arm64 emit，以后才加 RISC-V，而其他架构继续走 portable path。一份针对某个 emit 的 paper proof 不能自动覆盖其他 backend。普通 unit test 也很难暴露 weak-memory behavior、control-flow-hardening 错误或 metadata mismatch。

因此 native-operation interface 需要一份会跟着每个 backend implementation 走的 repeatable conformance artifact，而不是对整个 operation 给一个笼统的“verified”标签。

## 值得继续做成研究与生产系统的方向

### 1. 定义 verifier 可见、带 typed effect 的 native-operation contract

第一个 artifact 应该是一份 compact operation descriptor，而语义中心仍然是普通 BPF：

```text
operation = rotate64_v1
proof_sequence = [mov, lsh, rsh, or, ...]
effects = {
  regs: [r1 -> r0],
  memory: none,
  ordering: none,
  control_flow: fallthrough,
  faults: same_as(proof_sequence)
}
arch = x86_64
required_features = [ROL]
required_jit_properties = [ibt_safe]
native_emit_hash = sha256(...)
conformance = lean-proof:...
fallback = proof_sequence
```

现有 verifier 像检查普通 BPF 一样检查 proof sequence。一个很小的 registration checker 再检查 descriptor 是否自洽、effect class 是否属于 kernel 理解的固定集合，以及 selected backend 是否真的满足需要的 feature 与 hardening property。

Effect vocabulary 应该刻意保持小。纯 register transform、bounded memory transform、带明确 ordering 的 atomic、control-flow operation 可以分别有不同 proof obligation。放不进这些 effect class 的 operation，就继续保持 ordinary BPF，直到 contract 被有意识地扩展。

学术问题是：怎样设计一个足够表达常见 hardware idiom、又不会重新造出完整 machine-code verifier 的 effect system。生产价值是让 code review 有稳定边界：maintainer 能明确看到“这一条 native operation 到底多信任了什么”。

### 2. 把 cross-architecture differential + litmus conformance 做成 release gate

如果有精确 machine model，formal proof 很强，但 production interface 还需要覆盖不同 kernel、toolchain 与 CPU 的 regression evidence。

Native-operation conformance harness 应该从完全相同的 generated state 分别执行 proof sequence 与 native emit，再按声明的 effect set 比较结果。对纯 operation，可以做大规模 differential testing，覆盖 edge value、random state 与 verifier-derived range。对 memory / atomic operation，则加入 blitmus 风格的 concurrent litmus test，比较允许出现的 outcome，而不是只看最终数值。

同一套 harness 还可以检查 semantic proof 很容易遗漏的 architecture obligation：indirect target 是否有必须的 landing pad、emit 是否遵守 stack/clobber convention、是否绕过 JIT hardening、CPU feature 不支持时是否真的退回 portable fallback。

这套 matrix 可以进入 `selftests/bpf` 或类似 cross-architecture CI，在 x86-64、arm64、RISC-V、PowerPC 等 backend 上运行。每个 backend 都有自己的 conformance status。新增 native emit 只有在自己的 backend artifact 通过后才算完成。

有意思的研究问题是 formal evidence 与 hardware testing 应怎样组合，而不是假装一个可以替代另一个。Lean proof 可以证明模型里的 semantic relation；真实硬件上的 litmus 与 hardening test 可以找出模型漏掉的 assumption。两者不一致本身就是很有价值的结果。

### 3. 让 native implementation provenance 在运行时可见、可撤销

第三个 artifact 应记录实际执行的是哪一个 native implementation，而不仅是“哪个 BPF program 被加载”。

Program-info 或 JIT-info 可以暴露：

```text
bpf_prog_id = 4182
operation = rotate64_v1
operation_impl = x86_64/3
native_emit_hash = ...
proof_or_test_set = kops-einsn-2026.09
cpu_features = [bmi2, ibt]
jit_hardening = enabled
fallback_available = yes
```

这种 provenance 对 debugging 和 security response 都有用。如果某个 backend implementation 后来在 conformance test 里失败、kernel update 改变了相关 invariant，或者 CPU erratum 让之前 assumption 不再成立，就可以按 operation ID + backend version 禁用它。Program 直接退回 verifier-approved proof sequence，不需要重编 source，也不必继续运行已经可疑的 machine code。

这和昨天讨论的 profile invalidation 不是一回事。Workload 可以完全不变，失效的是 **“这个 machine implementation 值得信任”这条 claim**。

研究 prototype 应测量 disable/fallback latency、fallback 后性能损失、incident reproduction 需要多少 provenance，以及 operation-level revocation 是否可以明显比禁用整个 JIT backend 更简单。

## 哪些结果会改变这个判断？

有三类结果会削弱显式 native-operation contract 的必要性。

第一，更广泛的 evaluation 可能发现：考虑正常 compiler improvement 与 kernel JIT 优化后，operation-level native specialization 的收益很小。如果最终只剩少数 trivial arithmetic idiom 有明显收益，那么维护一整套 effect 与 conformance framework 可能还不如直接把这些 idiom upstream 到各 JIT。

第二，如果未来 major Linux architecture 上出现实用的 verified / translation-validated BPF-to-machine compiler，而且它已经覆盖 hardening、memory ordering 与 verifier-metadata obligation，那么 per-operation contract 的价值会下降，因为整个 backend 已经有一份持续维护的 machine-checked refinement proof。

第三，Linux 也可能演化出 generic JIT IR，把 verifier metadata、architecture capability、hardening constraint 与 machine lowering 都放进统一 pipeline。那样 native operation 可以成为 generic IR 里的普通 transformation，而不是独立注册的 emit。

当前证据更支持一个更小的近期 trust boundary。Kops 已经说明 verifier-visible proof sequence + native emit 能在不替换 verifier 的情况下恢复真实性能；近期 kernel JIT 改动说明 verifier metadata 与 architecture hardening 本来就需要显式协调；blitmus 则说明 cross-architecture correctness 可能依赖 single-thread value test 看不见的行为。**因此真正有用的 extension 不是“允许 module 发出任意更快的 machine code”，而是“允许一个有边界的 native implementation 替代 portable BPF contract，并把 assumption、effect、evidence 与 fallback 全部显式化”。**

## 参考资料

- IETF. [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), 2024 年 10 月。
- Linux kernel documentation. [`bpf_jit_harden` and BPF JIT sysctls](https://docs.kernel.org/next/admin-guide/sysctl/net.html), 访问于 2026-09-06。
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026。
- Yusheng Zheng, Hao Sun, Tong Yu. [kops and rejit: Safely Optimizing eBPF for Hardware and Workloads](https://lpc.events/event/20/contributions/2445/), Linux Plumbers Conference 2026 contribution，访问于 2026-09-06。
- Xu Kuohai et al. [bpf: Move constants blinding out of arch-specific JITs](https://lists.openwall.net/linux-kernel/2026/04/15/79), Linux kernel mailing list, 2026 年 4 月。
- Xu Kuohai et al. [bpf: Pass bpf_verifier_env to JIT](https://lists.openwall.net/linux-kernel/2026/04/16/307), Linux kernel mailing list, 2026 年 4 月。
- Linux kernel BPF maintainers. [Applied series: emit ENDBR/BTI instructions for indirect jump targets](https://lists.openwall.net/linux-kernel/2026/04/16/938), 2026 年 4 月。
- Greg Kroah-Hartman. [CVE-2026-64508: bpf: Support for hardening against JIT spraying](https://lists.openwall.net/linux-cve-announce/2026/07/25/203), 2026 年 7 月。
- Puranjay Mohan. [blitmus: Litmus-testing the eBPF memory model on real hardware](https://lpc.events/event/20/contributions/2432/), Linux Plumbers Conference 2026 contribution，访问于 2026-09-06。
- Martin Fink et al. [Proof-Carrying Verification for eBPF](https://lpc.events/event/20/contributions/2440/), Linux Plumbers Conference 2026 contribution，访问于 2026-09-06。