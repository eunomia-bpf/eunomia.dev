---
date: 2026-09-06
title: "eBPF 针对特定架构做优化后，还能保持可移植吗？"
description: "针对特定架构的 eBPF 快路径不能靠隐式假设。本文讨论能力协商、可移植 fallback 与跨 JIT 验证，定义机器特化仍能保持可移植性的系统契约。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - Compilers
  - Portability
  - Program Verification
research_question: "eBPF 怎样利用特定架构的 native instruction 与 JIT capability，同时不把一个可移植的 BPF 程序偷偷变成每台机器各自不同的契约？"
source_cutoff: 2026-09-06
status: daily-report
---

# eBPF 针对特定架构做优化后，还能保持可移植吗？

同一个 BPF 程序可以加载到 x86-64 服务器，也可以加载到 ARM64 机器，或者其他带 BPF JIT 的 Linux target。verifier 看到的仍然是 BPF instruction，但最终生成的 machine code 本来就不会一样。某个 backend 可能一条 native instruction 就能完成一个操作，另一个 backend 需要展开成多条指令，而且部分 JIT capability 本身就只存在于特定架构。

平时，这只是实现细节。可一旦 optimizer 主动依赖这些差异，它就会变成 portability 问题。

如果 userspace compiler 或 extension runtime 找到一个更快的 native implementation，不应该因此要求整个 BPF 程序退化成一个 opaque 的架构专用 binary。本文主张一个更窄的契约：**始终保留一个可移植的 BPF semantic witness，把每个 native specialization 都看成这个 witness 的可选、带 capability gate 的实现；目标机器不满足条件时，系统必须能够明确拒绝这条 fast path 或安全 fallback。**

<!-- more -->

这和上一篇[运行时画像驱动的 eBPF specialization](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/)不同。上一篇问的是 workload 或 deployment assumption 随时间变化后，一次 rewrite 还是否能被证明是同一个程序。本文可以假设 workload 完全稳定，只改变 JIT backend 或机器。

它也不同于[异构系统里的 eBPF 执行位置](https://eunomia.dev/zh/research/heterogeneous-ebpf-execution-placement/)。那篇讨论 kernel、userspace、NIC/DPU、GPU-side runtime 之间如何选 execution target。本文把 execution 继续留在 Linux BPF pipeline 里，只问 architecture-specific lowering 能否继续作为透明优化，而不是偷偷变成新的 application ABI。

## BPF ISA 本来就在区分可移植语义和实现能力

[RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) 标准化了 BPF ISA，并且明确引入 conformance group。所有 implementation 都必须支持 `base32`，同时可以选择支持 `base64`、`atomic32`、`atomic64`、`divmul32`、`divmul64` 等更多 group。RFC 也直接解释了这个设计的目的：runtime 和 compiler 可以通过 capability discovery 知道双方共享的是哪一组 BPF instruction semantics。

这个先例很重要。portability 并不等于每个 runtime 都必须实现所有 optional feature，而是 program 和 runtime 必须知道彼此同意的 semantic contract 是什么。

Linux 在 ISA 下面还有一层。kernel 文档列出了 x86-64、ARM、ARM64、PowerPC、RISC-V、s390、MIPS、SPARC 等多种 BPF JIT backend。它们把相同的、已经验证过的 BPF semantics 翻译成不同的 native instruction stream。

当前 Linux 的 [BPF Design Q&A](https://github.com/torvalds/linux/blob/master/Documentation/bpf/bpf_design_QA.rst) 给了一个很具体的例子。对于 32-bit ALU operation，某个架构如果通过 `bpf_jit_needs_zext()` 表示自己的 JIT 需要显式 zero-extension，verifier 就可以插入对应的 zext instruction；如果 backend 对其中一些 case 有硬件支持，还可以在本地 peephole 里去掉不必要的 extension。语义约束是共享的，lowering strategy 不是。

当前 kernel 的 [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c) 也用一组 weak architecture hook 表达类似关系，包括 `bpf_jit_needs_zext()`、`bpf_jit_supports_subprog_tailcalls()`、`bpf_jit_supports_kfunc_call()` 和 `bpf_jit_supports_far_kfunc_call()`。默认实现比较保守，具体架构再 override 自己真正支持的能力。

换句话说，architecture capability 早就在影响 BPF admission 和 lowering。缺少的是 optimizer 自己引入 native fast path 后，同样明确的 capability contract。

## Kops 说明 native fast path 可以保留可移植的 semantic witness

[Kops](https://arxiv.org/abs/2606.24213) 很适合观察这个边界，因为它不要求 verifier 去理解任意 machine code。一个 Kops operation 同时有两种形式：一段普通 BPF instruction 组成的 proof sequence，交给现有 verifier 检查；以及一个 architecture-specific native emit，供对应的 JIT path 使用。Lean 4 proof 再把 native implementation 和 proof sequence 的结果联系起来。

论文实现了 7 个 hardware idiom，在 x86-64 和 ARM64 上评估，microbenchmark 最高提升 24%，production application 最高提升 12%。仓库公开的 paper artifact 也明确写出了 per-architecture 差异：每个 module 只覆盖一个架构上的一条 native instruction，评估中的 module tree 在 x86-64 上有 14 个 module，在 ARM64 上有 11 个。

proof-sequence / native-emit 这个拆分本身就是很好的 portability primitive。它说明同一个 semantic operation 即使在某个 target 上有更好的 native implementation，也仍然可以用普通 BPF 表达。

但是在真正的 multi-architecture deployment 里，还要回答一些 local proof 之外的问题：这台机器到底允许选择哪个 native implementation？它依赖什么 kernel/JIT condition？当前架构根本没有对应实现时怎么办？loader 应该回到 proof sequence、跳过优化，还是拒绝整个 program？两台机器虽然用了不同 native implementation，operator 能不能确认它们执行的是同一个 semantic operation？

证明一段 native emit 是对的，并不会自动定义这套 deployment contract。

## Architecture specialization 应该经过协商，而不是藏在实现里

一个可移植的 optimizer 至少需要两层 capability 信息。

第一层是 BPF semantic contract，包括 ISA conformance group、program type、helper 或 kfunc dependency、verifier-visible effect，以及加载 portable form 所需的其他 platform requirement。

第二层才是 optional implementation contract。它说明某个 architecture-specific native specialization 可以替换哪一段 portable proof sequence，以及在哪些 machine/JIT 条件下才允许这样做。

可以先用类似下面的表示做研究原型：

```text
semantic_op = rotate64_v1
proof_seq_hash = sha256(portable_bpf_sequence)
required_bpf_groups = [base64]

native_impl = x86_64_rotate_v3
target_arch = x86_64
required_cpu_features = [...]
required_jit_capabilities = [...]
native_emit_hash = sha256(native_emit)
proof = lean4:rotate64_v1_x86_64_v3
fallback = proof_sequence
```

这些字段不是建议直接变成 Linux ABI。重点在于，native implementation 必须因为 eligibility predicate 被满足才被选中，而不是因为 optimizer 恰好在某种架构上编译过一次，之后大家就默认它在别的机器也成立。

这样一来，failure 的含义也会更清楚。一个 optimization unavailable 通常应该只是 optimization miss，而不是 application failure。ARM64 没有某个 x86-only implementation，不妨碍 loader 执行 portable proof sequence，或者选择一个被证明等价的 ARM64 implementation。

像 [bpftime](https://github.com/eunomia-bpf/bpftime) 这样的 userspace runtime 可以先在不修改 Linux userspace ABI 的前提下实验这层 negotiation。kernel verifier 仍然检查普通 BPF semantics，runtime 或 extension layer 决定当前 target 是否允许使用某个已经有 proof 的 native implementation。

## 现有研究还缺什么

### ISA conformance 还描述不了 optimizer-level native capability

RFC 9669 的 conformance group 解决的是 instruction-set interoperability。它不会告诉 loader：当前 kernel JIT 是否能把 optimizer 自己定义的 semantic operation 通过某条 native idiom 实现，也不会把这条 lowering 对应的 proof 和 fallback 一起表达出来。

Linux architecture hook 暴露了部分 backend capability，但它们是 kernel implementation interface，不是一个 portable optimizer 可以直接拿来解释 deployment decision 的统一 manifest。

因此现在中间仍有一个明显空档：`这个 BPF program 可以移植` 和 `这条优化实现可以在这里使用` 不是同一件事。

### Per-architecture native module 会自然分化，但缺少共同的 portability oracle

architecture-specific code 本来就会以不同速度演进。Kops 的 x86-64 和 ARM64 module 数量不同，是因为一些 operation 只针对一个架构，这本身没有问题。真正的问题是 deployment 如果默认 optimization coverage、性能和 fallback behavior 在不同 target 上都对称，就会把实现差异误当成系统契约。

一个 portable system 应该把 partial coverage 变成 first-class result。native operation 不支持时，必须有可观察的 fallback 或 eligibility result，而不是静默改变另一个机器上 program 的可用行为。

### 两个 backend 上有 speedup，还不等于 performance portability

优化论文通常报告支持机器上的 speedup。fleet operator 需要回答的是另一组问题：同一个 semantic BPF artifact 移动到不同架构、不同 kernel version 后，optimized implementation 还有多大覆盖率？不支持时损失多少性能？fallback 是否仍保持同一组 observable behavior？

没有这组 evidence，architecture specialization 完全可能让 benchmark 更快，却让 deployment 更不可预测。

## 有学术和生产价值的方向

### 1. 做一个两层 specialization capability manifest

第一个 artifact 应该让 semantic contract 和 native implementation contract 可以被独立查询。

semantic level 记录 portable BPF operation 或 proof sequence、conformance requirement、program type、effect scope 和稳定 identity。implementation level 再注册一个或多个 architecture-specific native implementation，并写清楚 eligibility predicate。

selection 可以变成一个很小的 negotiation protocol：

```text
portable semantics accepted?
        |
        +-- no  -> reject program
        |
        +-- yes -> find eligible native implementation
                     |
                     +-- found -> use proven fast path
                     |
                     +-- none  -> execute portable proof sequence
```

研究问题是怎样定义一套 capability vocabulary：它必须具体到足以阻止错误 selection，但又不能把每个 JIT 内部细节冻结成永久 ABI。生产价值则很直接，mixed fleet 和 rolling kernel upgrade 都可以得到可预测 fallback。

原型可以先完全做在 kernel ABI 之上，例如把 manifest 放进 ELF section 或 sidecar object，由 loader 或 re-JIT runtime 做 resolution，Linux verifier 继续作为 portable form 的最终 admission boundary。

### 2. 把一个 semantic operation 打包成多个带 proof 的 backend

第二个 artifact 应该让 specialization unit 从一开始就是 multi-backend 的。

不要把 `x86 implementation A` 和 `ARM implementation B` 当成两个互不相关的 optimization module，而是给一个 semantic operation identity 配上：

- 一份 portable BPF proof sequence；
- 零个或多个 x86-64 native emit；
- 零个或多个 ARM64 native emit；
- 未来可加入的 RISC-V、s390 或其他实现；
- 每个 native emit 对应的 proof 或 translation-validation result；
- 没有 native implementation 匹配时明确的 fallback rule。

这相当于把 Kops 很有价值的 proof/native split，从 local implementation safety 推进到 deployment portability。新增 backend 不需要改变 semantic operation ID，也不要求 application 重新分发另一套 source program。

关键实验应该故意制造 capability mismatch：给 loader 一个错误架构的 module、缺少 CPU feature 的 target、旧的 JIT capability set，以及改变 backend support 的 kernel upgrade。预期结果应该是 deterministic 地拒绝 native implementation，然后继续成功执行 portable form，而不是出现含糊的 selection behavior。

### 3. 把 portability 本身变成优化评估指标

第三个 artifact 应该是 cross-JIT benchmark，基本单元不是“一台机器上的一个优化”，而是“一个 semantic BPF workload 在一组 target matrix 上部署”。

使用同一组 XDP、tracing 和 compute-oriented BPF program，至少覆盖 x86-64 和 ARM64，再加入一个故意不提供某些 native operation 的 target。跨多个 kernel version 比较 stock JIT、architecture-specific specialization，以及带 negotiated fallback 的 specialization。

第一优先级 correctness oracle 仍然是 portable form 的 semantic equality，包括 return value、packet/context write、map effect 和 helper/kfunc-visible behavior。然后再测：

- 每个 architecture/kernel version 的 native specialization coverage；
- fallback rate 和原因；
- 相对 stock BPF 的性能；
- 相对当前 target 上 best safe implementation 的 cross-target performance regret；
- proof/check 和 selection overhead；
- 新增到 trusted computing base 的 native implementation 数量。

这样可以区分两种完全不同的结果。某个 optimization 在一台机器上快 20%，但在 fleet 其他机器上经常悄悄消失，performance portability 很弱。另一个 optimization 峰值 speedup 可能小一点，但支持时明确使用 fast path，不支持时明确 fallback，而且始终保留同一个 semantic artifact，deployment story 反而更强。

## 哪些结果会改变这个判断？

有三类结果会削弱单独设计 architecture-specialization contract 的必要性。

第一，stock Linux JIT 也许最终会吸收绝大多数有价值的 architecture-specific idiom，并且这些选择完全可以继续藏在 JIT 内部。如果 optimizer-added native operation 对真实 workload 没有额外的性能或 deployment 价值，那么再在 JIT 上面加 capability negotiation 只是在增加复杂度。

第二，BPF ISA conformance group 或未来的标准 capability interface 也可能扩展到足以直接表达这些 operation 和 runtime support。如果 compiler 已经可以通过稳定的标准机制协商所有有价值的 fast path，就不需要再造一层 optimizer-specific capability。

第三，大规模 cross-kernel 实验也可能发现，简单的 module presence 加 portable BPF fallback 在实践中已经够用。如果 architecture、CPU feature 和 JIT-version mismatch 从来不会导致含糊 selection 或 operational failure，更复杂的 manifest 可能不值得维护。

目前的 evidence 更支持相反方向。RFC 9669 已经把 capability discovery 当作 BPF interoperability 的一部分；Linux JIT backend 已经通过 architecture-dependent hook 表达实现差异；Kops 说明 hardware idiom 可以在保留 portable BPF proof sequence 的同时获得有意义的性能提升。**真正缺少的不是另一条 native instruction，而是一份 deployment contract，把这三件事连起来：这是什么 semantic operation，这台机器允许选择哪个 native implementation，以及 fast path 不存在时还剩下什么可移植行为。**

## References

- IETF. [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), 2024 年 10 月。
- Linux kernel documentation. [Linux Socket Filtering / BPF JIT compiler](https://kernel.org/doc/html/latest/networking/filter.html)，访问于 2026-09-06。
- Linux kernel documentation. [BPF Design Q&A](https://github.com/torvalds/linux/blob/master/Documentation/bpf/bpf_design_QA.rst)，访问于 2026-09-06。
- Linux kernel source. [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c)，访问于 2026-09-06。
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026。
