---
date: 2026-09-05
title: "eBPF 能用运行时画像做优化而不改变程序语义吗？"
description: "运行时画像可以指导更快的 BPF 重写，但通过 verifier 不等于语义等价。本文提出带 guard 的运行时 specialization contract。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - Profile Guided Optimization
  - Program Verification
  - Compilers
research_question: "eBPF runtime 能否利用 workload 与 deployment profile 在程序加载后继续做 specialization，同时证明可观测行为仍与原程序等价，并在假设过期时安全失效？"
source_cutoff: 2026-09-05
status: daily-report
---

# eBPF 能用运行时画像做优化而不改变程序语义吗？

一个 XDP 程序已经编译完成、通过 verifier，并在生产环境稳定运行了几个星期。现在 profile 告诉我们：某个分支几乎总是被走到，一个配置值几乎从不变化，而且当前 CPU 有一些 generic BPF bytecode 无法直接表达的指令。此时最自然的想法就是：既然已经知道真实 workload 长什么样，为什么不针对它重新优化一次？

难点并不是生成更快的代码，而是回答：**什么证据足以说明更快的代码仍然是同一个程序？**

Linux 给 BPF 提供了很强的安全边界。每个通过 `BPF_PROG_LOAD` 提交的 candidate 都要先经过 verifier，然后才进入 kernel JIT。但 verifier acceptance 和 optimization equivalence 是两个不同的问题。一个被重写后的程序可以 memory-safe、type-safe、bounded，也完全能通过 verifier，却仍然可能返回不同 verdict、更新不同 map entry、在不同条件下调用 helper，或者在 profile 从未见过的 rare input 上产生错误结果。

本文的核心判断是：运行时 specialization 需要两层 contract。kernel 继续决定 candidate 能不能安全执行；optimizer 需要另外证明 candidate 是否保持原程序的可观测行为。只要优化依赖 workload 或 deployment 的额外事实，这些事实还必须被显式表示、被 guard，并且可以失效。

<!-- more -->

这个问题不同于之前的[有状态 eBPF 原子升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/)。upgrade 本来就要改变程序或状态语义，所以重点是如何原子地切换 generation。runtime specialization 恰好相反：它声称新的 generation 只是同一语义的更快实现。它也不同于[异构系统里的 eBPF 执行位置](https://eunomia.dev/zh/research/heterogeneous-ebpf-execution-placement/)，后者讨论程序到底放在 kernel、userspace、NIC/DPU 还是其他 target。这里 target 可以完全不变，只是在同一个 BPF 语义下根据当前机器和 workload 重新生成实现。

## Verifier 证明 candidate 安全，但不会证明它与原程序等价

Linux [`bpf()` syscall 文档](https://docs.kernel.org/userspace-api/ebpf/syscall.html)把这个边界定义得很清楚。`BPF_PROG_LOAD` 负责 verify 并加载程序；`BPF_LINK_UPDATE` 可以替换一个现有 BPF link 上的程序；`BPF_ENABLE_STATS` 可以打开 BPF runtime statistics。这些 primitive 已经足够让一个 runtime 收集运行信息、生成新 candidate、重新送 verifier，并在不修改原应用 deployment flow 的情况下切换程序。

但这些接口都没有表达 old program 和 new program 之间的语义关系。

IETF 的 [BPF ISA 规范 RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html)定义了 32-bit truncation、signed arithmetic、atomic 等指令语义。这里的细节非常重要，因为正确的 rewrite 必须保持 BPF ISA 的语义，而不是只保持 optimizer 作者脑子里那个“应该等价的 C 程序”。RFC 9669 也明确提醒，verified BPF 到 machine instruction 的编译过程本身需要仔细审计，compiler 也可能引入漏洞。

因此 runtime specialization 最少要分开三个判断：

```text
kernel verifier:     这个 candidate 能安全执行吗？
optimizer checker:   这个 candidate 和原程序 contract 等价吗？
profile contract:    specialization 所依赖的假设现在还成立吗？
```

第一项通过，并不能推出第二项和第三项。

## 现有 BPF optimizer 已经证明 equivalence checking 是可做的

这并不是一个纯理论问题。已经有几类 BPF optimizer 把 semantic preservation 当成独立 obligation。

[K2](https://k2.cs.rutgers.edu/)用 program synthesis 优化 BPF bytecode，同时检查 correctness 与 verifier safety。它在 packet-processing workload 上能生成更小、更快的程序，但 equivalence checking 是 optimizer 自己必须完成的工作，并不是“反正 kernel verifier 会兜底”。

[EPSO](https://arxiv.org/abs/2511.15589)在 2025 年进一步把这个方向做成更快的 online path：昂贵的 superoptimization 放到 offline，得到的 rewrite rule 被 cache，之后可以快速应用到新的 BPF program。论文报告发现 795 条 rewrite rule，相比 Clang output 平均减少 24.37% program size，并平均降低 6.60% runtime。对本文更重要的是它的 architecture：一个可以重复使用的 rewrite rule 仍然需要明确的 equivalence argument，不能只因为 rewritten program 能过 verifier 就把它当成 transparent optimization。

[Kops](https://arxiv.org/abs/2606.24213)处理的是另一个边界。stock BPF JIT 为了保持简单和可信，通常接近一次 single-pass、逐 BPF instruction 的 translation。Kops 让一个 operation 同时携带 verifier 可见的普通 BPF proof sequence 和 hardware-specific native emit，再用 Lean 4 proof 把两者的结果联系起来。论文中的 7 个 hardware idiom 在 x86-64 和 ARM64 microbenchmark 上最高提升 24%，生产 application 最高提升 12%。

这些工作共同说明，BPF optimization 完全可以携带比“这个新程序过了 verifier”更强的证据。它们主要利用 program structure 或 hardware capability。runtime profile 带来的事实则不同：它可能只在当前 deployment、当前 workload phase 里成立。

## Runtime specialization 会把 observation 变成 assumption

Linux Plumbers Conference 2026 的公开 contribution [“kops and rejit: Safely Optimizing eBPF for Hardware and Workloads”](https://lpc.events/event/20/contributions/2445/)描述了 BpfReJIT：userspace LLVM runtime 可以拦截未修改应用的 BPF load/attach，基于 configuration、workload 和 kernel version 改写 bytecode，然后把每个 candidate 重新送回原来的 verifier/JIT。公开摘要直接把这种 runtime speculative optimization 类比到 V8 和 JVM。

这带来一个 static optimizer 不需要回答得这么明确的问题：**哪些 profile fact 只是 hint，哪些已经变成 generated code 的 semantic precondition？**

branch-frequency profile 通常可以安全地用于 basic-block layout，因为 cold path 仍然存在；hotness 变了最多让 performance 变差。runtime configuration value 就不同了。如果 optimizer 把一次 map lookup 直接折叠成 constant，那么它必须知道这个值在 specialization generation 存活期间不会变化，或者生成 guard，在值改变时 fallback。更加危险的是因为某个 helper path 在有限 trace 中一次都没出现，就直接删掉它。finite profile 的“没有看见”并不等于“永远不可能发生”。

成熟 speculative JIT 的核心也不是简单的 `profile -> faster code`，而是：

```text
profile -> assumptions -> optimized code + invalidation path
```

BPF 在这里有一个很有意思的优势：portable original program 天然就可以作为 deoptimization target。如果 assumption 失效，runtime 可以直接退回原程序，或者生成下一个 specialized generation；无论哪一种 candidate，kernel verifier 仍然是最终 safety boundary。

## Kernel JIT 本身也说明 transformation metadata 不能和代码脱节

2026 年 BPF kernel 的变化给了一个很具体的提醒：transformed code 不能只是一个匿名 machine-code blob。

一个已经进入 kernel 的 patch series 把 [constant blinding 从各 architecture-specific JIT 移到更通用的 verifier/JIT pipeline](https://lists.openwall.net/linux-kernel/2026/04/15/79)。原来的问题是 JIT 私有 instruction copy 被 rewrite 后，global `env->prog->insnsi` 和 `insn_aux_data` 仍然对应旧指令，导致 transformed instruction 与 verifier metadata 不同步。新的设计让 instruction rewrite 与相关 auxiliary state 一起更新。

另一个刚在 2026 年 9 月更新到 v9 的 patch series 给 [JITed BPF program 增加 KASAN check](https://lkml.iu.edu/2609.0/08000.html)。v9 cover letter 还专门修复了错误的 stack instrumentation，并增加额外条件避免对错误 stack offset 插桩。KASAN 本来就会为了检测 memory error 而改变执行行为，所以它不是 equivalence-preserving optimizer。这里有价值的是更窄的一点：一旦 JIT 开始注入和改写 machine-level behavior，architecture-specific transform、instruction identity、metadata 和 debug 信息必须一起维护。

如果 runtime 同时保留多个不同 profile epoch 对应的 specialized generation，这个要求只会更强。

## 现有工作还弱在哪里

### Verifier success 不是 transparent replacement certificate

runtime 可以拿原 BPF program 生成一个新程序，新程序可以成功通过 verifier，也可以成功通过 `BPF_LINK_UPDATE` 替换原 link。这整条路径都不能证明 packet verdict、return value、map write、tail call、helper effect 或其他 externally visible state transition 与原程序相同。

对局部 arithmetic rewrite，K2/EPSO 一类 SMT 或 synthesis-based equivalence checker 已经说明问题有解。whole-program BPF 更难，因为 helper、map、kernel context、concurrency 和 program-type-specific effect 都属于语义的一部分。因此 production re-JIT 不应该只有一个模糊的 `verified=true`，而应该说明它到底证明了什么 scope 的 equivalence。

### Profile 有 lifetime，但 optimized code 通常没有显式 lifetime

workload profile 是历史样本。branch bias 会反转；configuration map 会更新；VM migration 后 CPU feature 可能不同；kernel update 后 helper/kfunc capability 或 BTF layout 也可能变化。如果这些事实影响了 code generation，那么 running program 必须能机器可读地表达自己依赖了哪些事实。

否则一个 optimization 在生成时正确，十分钟之后就可能已经没有 justification。

### Operator 能看到 loaded program，却不一定知道为什么现在跑的是这个 generation

Linux 可以暴露 BPF program ID、link、metadata 和 JIT 相关信息，但 dynamic specialization 会制造新的 identity 问题：这个 candidate 是从哪个 original program 来的？哪个 profile epoch 触发了它？嵌入了哪些 assumption？做了哪些 transform？为什么 runtime 选择了 generation 17 而不是 16？

如果没有这层 provenance，performance regression 或 correctness incident 很难复现。incident 发生后 dump 出来的程序，可能已经不是处理出问题 workload phase 的那个 program。

## 值得实现的研究方向

### 1. 每个 specialized generation 都生成 optimization-equivalence certificate

第一个 artifact 是一个和 specialized generation 绑定的小型 certificate，同时标识 stable semantic source 与具体 candidate：

```text
source_prog_hash = sha256(original_bpf)
candidate_hash = sha256(specialized_bpf)
program_type = XDP
equivalence_scope = [return, packet_writes, map_effects, helper_trace]
transform_set = [branch_layout, alu_rewrite, const_fold]
checker = translation_validation_v3
checker_result = equivalent
kernel_verifier = accepted
profile_epoch = 418
assumptions = [config_generation=92]
```

这里故意把三件事拆开：candidate 通过 Linux verifier；optimizer 对某一个定义清楚的 equivalence relation 做了检查；任何让 rewrite 变成 conditional correct 的 assumption 另外列出。

checker 不需要只有一种实现。小 instruction slice 可以使用类似 K2/EPSO 的 SMT-backed translation validation；Kops-style hardware op 可以用 proof sequence 加 machine-level proof；helper-heavy region 如果 checker 不能建模，就保守地拒绝 optimization。`unknown` 是完全合理的结果，runtime 应该直接执行 original program。

学术问题是怎样把 BPF-specific effect 做成 compositional equivalence。生产价值则很直接：operator 可以知道 optimizer 到底证明了什么，而不需要无条件相信一个 opaque compiler。

### 2. 把 profile-derived fact 变成有 guard 的 specialization dependency

第二个 artifact 是和 active specialization generation 绑定的 assumption registry。

有些 optimization 根本不需要 runtime assumption。比如根据 branch frequency 重排 basic block，只要所有 edge 都保留，hotness 只是 performance hint。另一些 transformation 是 conditional：把 config map entry constant-fold 掉、针对 kernel capability specialization helper path、或者因为 feature flag 固定而删掉 case，都需要一个 validity condition。

runtime 应该明确区分这些 dependency：

```text
layout_hint(branch_17 = 99.8% taken)          -> no semantic guard
config(map_fd=8,key=3,generation=92,value=1) -> invalidate on generation change
kernel_btf(hash=...)                           -> invalidate on kernel/BTF change
cpu_features(avx2,bmi2)                        -> invalidate on migration
```

一旦 dependency 被违反，就触发 bounded deoptimization：把 attachment 切回 portable original，或者切到另一个已经 verified 的 generation，然后再决定是否 recompile。`BPF_LINK_UPDATE` 可以是实现这个动作的 primitive，但研究贡献不是“怎么换 link”。真正的问题是：怎样把一个 profile assumption 和“这个 optimized generation 在哪段时间里仍然有语义依据”绑定起来。

这样 BPF 可以借鉴 speculative language runtime 最有用的一部分，而不需要复制整套 JVM/V8 execution model。portable BPF bytecode 已经是天然 fallback，kernel verifier 也已经是每次 rewrite 后的 safety gate。

### 3. 用 phase shift 和 rare-path counterexample 测 specialization

profile-guided optimizer 在 stationary benchmark 上很容易看起来非常好，然后在第一次 workload phase change 时出错。评测应该把 profile staleness 当作测试对象，而不是当成 noise 平均掉。

一个有区分度的 benchmark 可以拿同一批真实 BPF program，人工控制每次只改变一种 assumption：packet mix、configuration-map generation、kernel/BTF version、CPU capability、rare error path、map pressure、helper outcome。每次运行同时保留 reference program 与所有 specialized generation。

至少比较四类 implementation：

- ordinary Clang + stock kernel verifier/JIT；
- K2/EPSO 一类 static equivalence-preserving optimization；
- Kops 一类 hardware-specialized operation；
- 带 guard 与 deoptimization 的 runtime profile-guided re-JIT。

主要 correctness metric 应该是 **adversarial phase change 下相对 portable reference 的 observable divergence**，包括错误 return value、packet mutation、map effect 和 helper-effect trace。其次才比较 speedup、invalid assumption detection time、deoptimization latency、equivalence-check cost、verifier cost、specialization churn，以及有多少 proposed optimization 被保守拒绝。

测试中还必须专门放入 training profile 从没出现过的 rare path。如果 optimizer 把“没观察到”当成“不可能”，benchmark 应该立即让这个错误暴露出来。

为了 debug，每一个 result 还要记录 source program hash、specialization generation、profile epoch、transform set、assumption set 和最终 JIT identity。没有这层 provenance 的 performance number 很难复现 dynamic optimizer。

## 什么结果会改变这个判断？

有三类结果会削弱单独设计 runtime specialization contract 的必要性。

第一，如果实验发现真正有收益的 workload-guided BPF optimization 几乎全部都是 unconditional transform，例如只根据 profile 调整 layout 和 scheduling，而不删除 semantic path，那么 static equivalence checking 可能已经足够，assumption/deoptimization layer 的收益就很小。

第二，如果 production evaluation 发现把 modern Clang、static BPF optimizer、verifier constraint 和 stock JIT 都算进去之后，profile-guided re-JIT 只有很小收益，那么额外引入 compiler、profile collector、equivalence checker 和 generation manager 就不值得。

第三，如果 Linux 最终提供标准化 optimization 或 translation-validation interface，能够把 transformed BPF / machine code 与 verifier-visible semantics 绑定，并直接暴露 generation provenance；如果这个接口还能表达 conditional assumption 与 invalidation，那么 userspace 再维护一层 certificate 就会变得重复。

当前证据更支持相反方向。K2 和 EPSO 说明 BPF bytecode 里仍有实际可利用的 semantics-preserving optimization 空间；Kops 说明 hardware-specific operation 可以在显式 proof structure 下拿回性能；BpfReJIT 的公开设计则说明 deployment/workload information 可以在不替换 kernel verifier 的前提下进入 re-JIT。**现在缺的并不是更多 profile 数据，而是一份可以说明“为什么这个 specialized generation 仍然是同一个程序，以及这个结论什么时候过期”的 contract。**

## 参考资料

- IETF. [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), 2024 年 10 月。
- Linux kernel documentation. [`bpf()` syscall reference](https://docs.kernel.org/userspace-api/ebpf/syscall.html), 访问于 2026-09-05。
- Q. Xu et al. [K2: Synthesizing Safe and Efficient Kernel Extensions for Packet Processing](https://k2.cs.rutgers.edu/), SIGCOMM 2021。
- Qian Zhu et al. [EPSO: A Caching-Based Efficient Superoptimizer for BPF Bytecode](https://arxiv.org/abs/2511.15589), 2025。
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026。
- Yusheng Zheng, Hao Sun, Tong Yu. [kops and rejit: Safely Optimizing eBPF for Hardware and Workloads](https://lpc.events/event/20/contributions/2445/), Linux Plumbers Conference 2026 contribution，访问于 2026-09-05。
- Xu Kuohai et al. [bpf: Move constants blinding out of arch-specific JITs](https://lists.openwall.net/linux-kernel/2026/04/15/79), Linux kernel mailing list，2026 年 4 月。
- Alexis Lothoré. [bpf: add support for KASAN checks in JITed programs, v9](https://lkml.iu.edu/2609.0/08000.html), Linux kernel mailing list，2026 年 9 月。
